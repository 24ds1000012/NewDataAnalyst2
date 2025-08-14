# =========================
# agent.py (merged & optimized)
# =========================

# --- Core imports ---
import os
import re
import json
import asyncio                                   # CHANGE: async helpers
import logging
import base64
import tempfile
from io import BytesIO, StringIO
from functools import lru_cache                   # CHANGE: caching
from typing import Dict, List, Tuple, Any

# --- Data / math / viz ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- LLM ---
from dotenv import load_dotenv
from openai import OpenAI

# --- Web / scraping ---
import requests
from bs4 import BeautifulSoup
import certifi

# --- ML / utils ---
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer, util

# --- Parsing / files ---
import duckdb
import pdfplumber
import pytesseract

# --- Fuzzy matching ---
from fuzzywuzzy import fuzz  # (kept for compatibility)

load_dotenv()

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# OpenAI client
# ---------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ---------------------------
# Global constants
# ---------------------------
MAX_ATTEMPTS = 4

# =========================
# Models & Heavy Resources
# =========================

# CHANGE: cache sentence model once at startup
_SENTENCE_MODEL = None
try:
    _SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Loaded SentenceTransformer 'all-MiniLM-L6-v2'")
except Exception as e:
    logger.error(f"SentenceTransformer load failed: {e}", exc_info=True)
    _SENTENCE_MODEL = None

# CHANGE: persistent DuckDB connection with httpfs/parquet preloaded
@lru_cache(maxsize=1)
def _get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    duckdb_dir = os.getenv("DUCKDB_HOME", "/tmp/duckdb")
    os.makedirs(duckdb_dir, exist_ok=True)
    db_path = f"{duckdb_dir}/duckdb.db"
    logger.info(f"[DuckDB] Using database at: {db_path}")
    con = duckdb.connect(database=db_path)
    con.execute(f"SET temp_directory='{os.getenv('DUCKDB_TEMP_DIR', '/tmp/duckdb')}'")
    con.execute("SET threads=4")  # CHANGE: allow parallelism
    con.execute("SET http_timeout=30000")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if aws_access_key and aws_secret_key:
        con.execute(f"SET s3_access_key_id='{aws_access_key}'")
        con.execute(f"SET s3_secret_access_key='{aws_secret_key}'")
    con.execute("SET s3_region='ap-south-1'")
    return con

# =========================
# Regex (compiled) & helpers
# =========================

# Strip any leading non-digit characters (e.g., letters like 'T' in 'T$2,257,844,554')
_RE_LEADING_ALPHA = re.compile(r'^[^\d]+')

# Remove currency symbols, commas, spaces, percent signs
_RE_STRIP_SYMBOLS = re.compile(r'[\$₹€,%\s]')

# Keep only numeric characters, decimal points, or negative signs
_RE_NUM_KEEP = re.compile(r'[^\d.e-]')

# Fix trailing commas in JSON strings
_RE_JSON_TRAIL_COMMA = re.compile(r',\s*([}\]])')

# Map superscript digits to ASCII
_SUP_DIGITS = str.maketrans({
    '⁰':'0','¹':'1','²':'2','³':'3','⁴':'4',
    '⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9', 'ᵀ': '', 'ᵀ': '','ˢ': '','ᴹ': ''  
})

# Mapping of superscript digits to ASCII digits (kept for clarity)
SUPERSCRIPT_DIGIT_MAP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', 'ᵀ': '','ˢ': '','ᴹ': ''  
}

def normalize_superscripts(value: Any) -> str:
    """Convert superscripts to ASCII using precompiled map."""
    if not isinstance(value, str):
        value = str(value)
    if any(ord(c) in range(0x2070, 0x209F) or ord(c) in [0x00B2,0x00B3,0x00B9] or c == 'ᵀ' for c in value):
        logger.info(f"Superscripts detected in '{value}'")
    return value.translate(_SUP_DIGITS)

def _scale_word_to_multiplier(s: str) -> float:
    s = s.lower()
    if 'billion' in s or 'bn' in s:
        return 1e9
    if 'million' in s or 'mn' in s:
        return 1e6
    if 'crore' in s or 'cr' in s:
        return 1e7
    if 'lakh' in s:
        return 1e5
    return 1.0

def clean_numeric_value(value: Any) -> float:
    """Cleans a numeric string and converts to float."""
    try:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        
        s = normalize_superscripts(str(value).lower().strip())
        s = _RE_LEADING_ALPHA.sub('', s)           # remove leading letters
        s = _RE_STRIP_SYMBOLS.sub('', s)          # strip $, %, commas, etc.
        mult = _scale_word_to_multiplier(s)       # scale for million/billion/crore
        
        s = _RE_NUM_KEEP.sub('', s)               # keep only digits, dot, minus
        return float(s) * mult if s else np.nan
    except Exception as e:
        logger.warning(f"Failed to clean '{value}', returning NaN: {e}")
        return np.nan
    
# CHANGE: vectorized version for pandas Series
def clean_numeric_series(series: pd.Series) -> pd.Series:
    return series.map(clean_numeric_value)

def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_cols, categorical_cols, temporal_cols = [], [], []
    date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d']
    for col in df.columns:
        sample = df[col].dropna().head(10)
        if len(sample) < 2:
            categorical_cols.append(col)
            continue

        col_l = str(col).lower()
        if fuzz.partial_ratio('date', col_l) > 60:
            temporal_cols.append(col)
            continue
        if any(fuzz.partial_ratio(k, col_l) > 60 for k in ['name','country','state','title','symbol','company']):
            categorical_cols.append(col)
            continue

        # CHANGE: Fix 'notna' error by using pd.Series instead of numpy array
        cleaned = [clean_numeric_value(v) for v in sample]
        numeric_like = pd.Series(cleaned)
        if numeric_like.isna().sum() <= max(1, int(len(sample)*0.3)):
            numeric_cols.append(col)
            continue

        was_temporal = False
        for fmt in date_formats:
            t = pd.to_datetime(sample, format=fmt, errors='coerce')
            if t.notna().sum() >= max(1, int(len(sample)*0.7)):
                temporal_cols.append(col)
                was_temporal = True
                break
        if not was_temporal:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols, temporal_cols

# =========================
# Column mapping utility
# =========================

# CHANGE: reusable mapper for generated code; cached per DF signature
def _df_signature(df: pd.DataFrame) -> str:
    return '|'.join(map(str, df.columns)) + f"::{len(df)}"

@lru_cache(maxsize=128)
def _cached_map_columns(sig: str, targets_key: str) -> Dict[str, str]:
    # memo only by signature+targets (real work in wrapper)
    return {}

def map_columns(df: pd.DataFrame, required_columns, use_sort: bool=False) -> dict:
    """
    Maps required columns to actual DataFrame columns using fuzzy matching.
    
    Args:
        df: pandas DataFrame
        required_columns: 
            - dict of target -> list of possible keywords
              Example:
              {
                'id': ['name','company','symbol','title'],
                'metric': ['change','% change','price','gross','worldwide gross'],
                'date': ['date','year','release']
              }
            - OR list of required column names (exact or close matches will be found)
        use_sort: unused, kept for backward compatibility
    Returns:
        dict: target -> matched column name (or None if no match)
    """

    # Allow list input: convert to dict form
    if isinstance(required_columns, list):
        required_columns = {col: [col] for col in required_columns}
        logger.info(f"[map_columns] Converted list to dict: {required_columns}")

    df_cols = [str(c) for c in df.columns]
    logger.info(f"df contains {df_cols}")
    logger.info(f"Required columns {required_columns.items()}")

    mapping = {}

    for target, keywords in required_columns.items():
        # Ensure keywords is a list
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, (list, tuple)):
            keywords = list(keywords)

        # Prefilter columns containing any keyword
        candidates = [c for c in df_cols if any(k.lower() in c.lower() for k in keywords)]

        # Fallback to fuzzy matching
        best_col, best_score = None, -1
        scan_cols = candidates if candidates else df_cols
        for c in scan_cols:
            score = max(fuzz.partial_ratio(k.lower(), c.lower()) for k in keywords)
            if score > best_score:
                best_score, best_col = score, c

        # Semantic fallback (optional)
        if (best_score < 50) and (_SENTENCE_MODEL is not None):
            try:
                emb_cols = _SENTENCE_MODEL.encode(df_cols, convert_to_tensor=True, normalize_embeddings=True)
                emb_kw = _SENTENCE_MODEL.encode([' '.join(keywords)], convert_to_tensor=True, normalize_embeddings=True)
                sims = util.cos_sim(emb_kw, emb_cols).cpu().numpy()[0]
                idx = int(np.argmax(sims))
                if float(sims[idx]) > 0.5:
                    best_col, best_score = df_cols[idx], int(sims[idx]*100)
            except Exception as e:
                logger.warning(f"Semantic fallback failed: {e}")

        # Assign mapping
        mapping[target] = best_col
        if best_col:
            logger.info(f"[map_columns] {target} -> '{best_col}' (score~{best_score})")
        else:
            logger.warning(f"[map_columns] No match for {target}")

    return mapping

# =========================
# OpenAI call
# =========================

async def ask_gpt(messages, model="gpt-4o-mini", temperature=0):
    try:
        # CHANGE: call in a thread so event loop isn’t blocked
        def _call():
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            ).choices[0].message.content.strip()
        return await asyncio.to_thread(_call)
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise

# =========================
# Code extraction & execution
# =========================

def extract_code_blocks(response: str) -> List[str]:
    # CHANGE: also allow ``` without language and strip backticks
    blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```(.*?)```", response, re.DOTALL)
    return [b.strip() for b in blocks]

# CHANGE: optional persistent selenium (reused) – created lazily
_SELENIUM_CTX = {"driver_path": None}

def init_selenium():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.core.os_manager import ChromeType

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.binary_location = '/usr/bin/chromium'

    # CHANGE: cache driver path (avoids repeated downloads)
    if not _SELENIUM_CTX["driver_path"]:
        _SELENIUM_CTX["driver_path"] = ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
    service = Service(_SELENIUM_CTX["driver_path"])
    driver = webdriver.Chrome(service=service, options=options)
    logger.info("Selenium WebDriver initialized.")
    return driver

# CHANGE: expose helpers to generated code & run exec in worker thread
async def safe_execute(code_blocks: List[str], global_vars: Dict[str, Any]):
    global_vars.update({
        'pd': pd, 'np': np, 'plt': plt, 'base64': base64,
        'BytesIO': BytesIO, 'StringIO': StringIO, 'json': json, 're': re,
        'logging': logging, 'os': os, 'requests': requests, 'BeautifulSoup': BeautifulSoup,
        'certifi': certifi, 'fuzz': fuzz, 'util': util, 'logger': logger,
        'clean_numeric_value': clean_numeric_value, 'clean_numeric_series': clean_numeric_series,
        'infer_column_types': infer_column_types, 'map_columns': map_columns,
        '_SENTENCE_MODEL': _SENTENCE_MODEL
    })

    # pre-inject duckdb & connection if needed by any block
    needs_duckdb = any(('duckdb' in b) or ('read_parquet' in b.lower()) for b in code_blocks)
    if needs_duckdb:
        con = _get_duckdb_connection()
        global_vars['duckdb'] = duckdb
        global_vars['con'] = con

    # add optional libs on demand
    for idx, code in enumerate(code_blocks):
        try:
            logger.info(f"Executing block {idx + 1}:\n{code[:1000]}{'...' if len(code)>1000 else ''}")

            needs_selenium = any(k in code.lower() for k in ['webdriver.', 'selenium.webdriver', 'chromedriver', 'webdriver_manager'])
            if needs_selenium:
                driver = init_selenium()
                global_vars['driver'] = driver

            if 'pdfplumber' in code:
                global_vars['pdfplumber'] = pdfplumber
            if 'pytesseract' in code:
                global_vars['pytesseract'] = pytesseract
            if 'LinearRegression' in code:
                global_vars['LinearRegression'] = LinearRegression

            # CHANGE: run exec in a worker thread to avoid blocking
            await asyncio.to_thread(exec, code, global_vars)

            if 'df' in global_vars and isinstance(global_vars['df'], pd.DataFrame):
                if global_vars['df'].empty:
                    logger.error("DataFrame is empty after loading.")
                    return False, "Empty DataFrame loaded."
                logger.info(f"Loaded DataFrame with columns: {list(map(str, global_vars['df'].columns))}")
            else:
                # not all code blocks must yield df, but your pipeline expects it after step code
                logger.info("No DataFrame yet from this block (may be expected for planning/setup).")
        except Exception as e:
            logger.error(f"Code block {idx + 1} failed: {e}")
            return False, str(e)
        finally:
            # CHANGE: keep driver warm for reuse; if you prefer closing each time, uncomment below
            # if 'driver' in global_vars:
            #     try:
            #         global_vars['driver'].quit()
            #     except Exception:
            #         pass
            pass
    return True, None

# =========================
# Cached fetchers (faster IO)
# =========================

# CHANGE: cache HTML fetch + table parse
@lru_cache(maxsize=64)
def _cached_read_html(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, verify=certifi.where(), timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))

# CHANGE: cache PDF->text for speed
@lru_cache(maxsize=32)
def _cached_pdf_text(path: str) -> str:
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if not t:
                try:
                    t = pytesseract.image_to_string(page.to_image().original) or ""
                except Exception:
                    t = ""
            text_chunks.append(t)
    return "\n".join(text_chunks)

# =========================
# Error-driven regeneration
# =========================

async def regenerate_with_error(messages, error_message, stage="step"):
    error_guidance = error_message
    if "HTTPConnectionPool" in error_message or "timeout" in error_message.lower():
        error_guidance += (
            "\nSelenium timed out. Increase WebDriverWait to 20s and use EC.presence_of_element_located((By.TAG_NAME, 'table')). "
            "Add requests.get(..., timeout=30)."
        )
    if "could not convert string to float" in error_message:
        error_guidance += (
            "\nApply numeric cleaning only to intended numeric columns using clean_numeric_value() from agent.py. "
            "Handle unexpected prefixes like 'S' or 'T' by stripping non-numeric leading characters. "
            "Check for malformed values like 'S1922598800' and log them."
        )
    if "expected x and y to have same length" in error_message:
        error_guidance += ("\nAlign series with df[[x,y]].dropna() before regression/correlation.")
    if "index out of bounds" in error_message or "no rows" in error_message:
        error_guidance += ("\nCheck for empty data after filtering; provide safe fallbacks.")
    if "No variable `result` or `results` found" in error_message:
        error_guidance += ("\nAssign final output to `result`.")
    if "print_png" in error_message and "optimize" in error_message:
        error_guidance += ("\nRemove optimize from plt.savefig; control size via dpi.")
    if "no tables found" in error_message.lower():
        error_guidance += ("\nTry Selenium with ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).")
    if "session not created" in error_message.lower() or "chromedriver" in error_message.lower():
        error_guidance += ("\nEnsure ChromeDriverManager matches Chromium and set options.binary_location.")
    if "no module named 'webdriver_manager.utils'" in error_message.lower():
        error_guidance += ("\nUse 'from webdriver_manager.core.os_manager import ChromeType'.")
    if "unable to obtain driver for chrome" in error_message.lower():
        error_guidance += ("\nUse webdriver_manager to install/manage ChromeDriver automatically.")
    if "column" in error_message.lower() or "key" in error_message.lower():
        error_guidance += (
            "\nDon’t assume exact names; use fuzzy match. Don’t clean categorical columns."
        )
    if "julianday does not exist" in error_message.lower():
        error_guidance += ("\nCompute date diffs in pandas, not DuckDB julianday.")
    if "failed to create directory" in error_message.lower() or "permission denied" in error_message.lower():
        error_guidance += ("\nEnsure DUCKDB_HOME exists; set temp_directory explicitly.")
    if "name 'model' is not defined" in error_message.lower():
        error_guidance += ("\nGuard model usage; fallback when insufficient data.")
    if "name 'Service' is not defined" in error_message.lower():
        error_guidance += ("\nImport and add Service to globals when using Selenium.")
    if "no module named 'pdfplumber'" in error_message.lower():
        error_guidance += ("\nInstall/import pdfplumber and inject into globals.")
    if "no module named 'pytesseract'" in error_message.lower():
        error_guidance += ("\nInstall/import pytesseract and inject into globals.")
    if "Can only use .str accessor with string values" in error_message:
        error_guidance += ("\nUse .astype(str) cautiously; prefer type-aware cleaning.")
    if "'int' object has no attribute 'lower'" in error_message.lower() or "object of type 'int' has no len()" in error_message.lower():
        error_guidance += ("\nEnsure df.columns are strings before fuzzy matching.")

    messages.append({
        "role": "user",
        "content": (
            f"The previous {stage} failed with this error:\n\n{error_guidance}\n\n"
            "Regenerate the {stage}. Inspect df columns/dtypes/head. Use fuzzy matching with df.columns cast to str. "
            "Map required columns via map_columns(); prefer numeric cleaning only on mapped numeric fields. "
            "For S3 Parquet, use DuckDB with hive_partitioning and restricted subsets. "
            "For plots, ensure base64 < 100kB using format='png', small figsize, and dpi tweaks. "
            "Analyze the user's question to determine the requested output format (e.g., JSON array of strings or JSON object with specific keys). "
            "Return `result` as the requested format, answering all questions in the order asked. "
            "Assign the final output to `result`."
        )
    })
    return await ask_gpt(messages)

# =========================
# JSON repair helper
# =========================

# CHANGE: add repair_json used in your final stage
def repair_json(s: str) -> str:
    # naive fixes for common JSON issues
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`")
    s = _RE_JSON_TRAIL_COMMA.sub(r"\1", s)  # remove trailing commas before ] or }
    # ensure quotes are double
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    return s

# =========================
# Main orchestration
# =========================

async def process_question(question: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Data Analyst Agent tasked with answering arbitrary analysis questions by generating Python code. "
                "Assume unknown data sources (web, DuckDB/S3 Parquet, local files). Avoid loading huge partitions; prefer hive partitions and selective queries. "
                "For web tables use requests + certifi + pandas.read_html(StringIO(...)). If no tables found, fall back to Selenium. "
                "Convert df.columns to str before fuzzy matching. Use fuzzy matching (>40) and, if available, semantic fallback via _SENTENCE_MODEL. "
                "Use map_columns(df, required_columns) to get a dict mapping target to column names and log the output of map_columns."
                "Preserve categoricals, parse temporals with pd.to_datetime(errors='coerce'). "
                "For plots, use matplotlib with figsize=(4,3), dpi<=100, bbox_inches='tight' to keep base64 < 100kB. "
                "Always produce and use map_columns(df, required_columns, use_sort=False). Use only mapped names for operations. "
                "Preserve categoricals and parse temporal columns. Clean numeric columns robustly (currency, percent, scale words, superscripts) "
                "Use matplotlib with small figsize/dpi, and return base64 under 100kB. "
                "Analyze the user's question to determine the requested output format (e.g., JSON array of strings or JSON object with specific keys). "
                "Return `result` as the requested format, answering each question in the order asked. "
                "Always assign final output to `result`."
            )
        }
    ]

    # Attachment parsing (kept)
    file_path = None
    if "Attachments:" in question:
        try:
            lines = [line.strip() for line in question.split('\n') if line.strip()]
            attachment_start = -1
            for i, line in enumerate(lines):
                if line.startswith("Attachments:"):
                    attachment_start = i
                    break
            if attachment_start == -1:
                return {"error": "No valid attachment section", "details": "Missing 'Attachments:' header"}
            attachment_details = lines[attachment_start + 1:]
            if not attachment_details:
                return {"error": "No file path provided", "details": "Attachment details are empty"}
            match = re.match(r"([^:]+):\s*(.+)", attachment_details[0])
            if not match:
                return {"error": "Invalid attachment format", "details": f"Expected 'filename: path', got '{attachment_details[0]}'"}
            _, path = match.groups()
            file_path = path.strip()
            if not os.path.exists(file_path):
                return {"error": "File not found", "details": f"No file exists at {file_path}"}
        except Exception as e:
            return {"error": "Failed to extract file path", "details": str(e)}

    messages.append({
        "role": "user",
        "content": (
            f"Analyze and break down this task into clear steps: {question}. "
            f"{'The question includes an attachment with file path: ' + file_path if file_path else 'No attachments provided; assume inline data or external sources.'} "
            "Identify the source and how to fetch it; use selective S3/DuckDB reads; inspect and clean data; map columns with map_columns."
        )
    })

    # --- Plan ---
    task_plan = await ask_gpt(messages)
    logger.info("Task Breakdown:\n" + task_plan)
    messages.append({"role": "assistant", "content": task_plan})

    global_vars: Dict[str, Any] = {
        "__name__": "__main__",
        # CHANGE: expose caches/helpers to generated code
        "_cached_read_html": _cached_read_html,
        "_cached_pdf_text": _cached_pdf_text
    }

    # --- Step code loop ---
    step_attempt = 0
    while step_attempt < MAX_ATTEMPTS:
        step_attempt += 1
        messages.append({
            "role": "user",
            "content": (
                "Write Python code to fetch and preprocess the data based on the task breakdown. "
                f"{'The question specifies a PDF/Excel/Word file at path: ' + file_path if file_path else 'No attachments provided; check for inline data or scrape as needed.'} "
                "Prefer selective S3/DuckDB reads with hive_partitioning=True. For web, use requests+read_html(StringIO). If none, fall back to Selenium. "
                "Print df columns/dtypes/head for debugging. Store the DataFrame in `df` and set `global_vars['df']`."
            )
        })
        code_response = await ask_gpt(messages)
        logger.info(f"[Attempt {step_attempt}] Step Code:\n{code_response}")

        code_blocks = extract_code_blocks(code_response)
        success, error = await safe_execute(code_blocks, global_vars)

        if success and isinstance(global_vars.get('df'), pd.DataFrame):
            break
        elif step_attempt < MAX_ATTEMPTS:
            code_response = await regenerate_with_error(messages, error or "No DataFrame created", "step code")
            logger.info(f"[Regenerated Attempt {step_attempt + 1}] Step Code:\n{code_response}")
        else:
            return {"error": "Step code execution failed after max attempts", "details": error or "Unknown"}

    # --- Dataframe metadata & inference ---
    metadata_info = "No dataframe created."
    if "df" in global_vars and isinstance(global_vars["df"], pd.DataFrame):
        try:
            df = global_vars["df"]
            buf = StringIO()
            df.info(buf=buf)
            buf.seek(0)
            metadata_info = buf.getvalue()
            numeric_cols, categorical_cols, temporal_cols = infer_column_types(df)
            metadata_info += f"\nInferred Numeric Columns: {numeric_cols}"
            metadata_info += f"\nInferred Categorical Columns: {categorical_cols}"
            metadata_info += f"\nInferred Temporal Columns: {temporal_cols}"
            metadata_info += "\nSample data (first 5 rows):\n" + str(df.head(5))
        except Exception as e:
            metadata_info = f"Error retrieving DataFrame metadata: {str(e)}"
    logger.info(f"DataFrame Metadata:\n{metadata_info}")

    messages.append({
        "role": "user",
        "content": (
            f"The dataframe metadata is:\n{metadata_info}\n\n"
            "Generate Python code to answer the question using `df`. "
            "Use map_columns to select columns; clean numeric fields; parse temporals; use df[[x,y]].dropna() for stats. "
            "Plot small (figsize=(4,3), dpi<=100) and return base64 under 100kB. "
            "Analyze the user's question to determine the requested output format (e.g., JSON array of strings or JSON object with specific keys). "
            "Return `result` matching the requested format, answering all questions in the order asked."
        )
    })

    # --- Final code loop ---
    final_attempt = 0
    while final_attempt < MAX_ATTEMPTS:
        final_attempt += 1
        final_code = await ask_gpt(messages)
        logger.info(f"[Final Attempt {final_attempt}] Final Code:\n{final_code}")
        final_blocks = extract_code_blocks(final_code)
        success, error = await safe_execute(final_blocks, global_vars)
        if success:
            break
        elif final_attempt < MAX_ATTEMPTS:
            final_code = await regenerate_with_error(messages, error or "Execution failed", "final code")
            logger.info(f"[Regenerated Attempt {final_attempt + 1}] Final Code:\n{final_code}")
        else:
            return {"error": "Final result code execution failed after max attempts", "details": error or "Unknown"}

    # --- Extract result ---
    try:
        result = global_vars.get("results", global_vars.get("result"))
        if result is None:
            raise ValueError("No variable `result` or `results` found.")
        if isinstance(result, str):
            try:
                result = json.loads(repair_json(result))
            except json.JSONDecodeError as e:
                return {"error": "Result is a string but not valid JSON", "details": str(e)}
        # make JSON-safe
        def to_json_safe(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, (list, tuple)):
                return [to_json_safe(x) for x in obj]
            if isinstance(obj, dict):
                return {k: to_json_safe(v) for k, v in obj.items()}
            return obj
        result = to_json_safe(result)
        json.dumps(result)  # validate
        logger.info(f"Final result: {result}")
        return result
    except Exception as e:
        logger.error(f"Result extraction failed: {e}")
        return {"error": "Result extraction failed", "details": str(e)}
