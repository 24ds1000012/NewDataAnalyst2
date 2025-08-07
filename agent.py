import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO, StringIO
import json
import re
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from json_repair import repair_json
import requests
from bs4 import BeautifulSoup
import certifi
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
from sklearn.linear_model import LinearRegression
import duckdb
import pdfplumber
import tempfile
import pytesseract
import tenacity
from selenium.common.exceptions import WebDriverException, TimeoutException
from fuzzywuzzy import fuzz

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

MAX_ATTEMPTS = 5

def initialize_duckdb():
    duckdb_dir = os.getenv("DUCKDB_HOME", "/tmp/duckdb")
    try:
        logger.info(f"Initializing DuckDB with database path: {duckdb_dir}/duckdb.db")
        os.makedirs(duckdb_dir, exist_ok=True)
        con = duckdb.connect(database=f"{duckdb_dir}/duckdb.db")
        con.execute(f"SET temp_directory='{os.getenv('DUCKDB_TEMP_DIR', '/tmp/duckdb')}'")
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
    except Exception as e:
        logger.error(f"Failed to initialize DuckDB: {e}", exc_info=True)
        raise

async def ask_gpt(messages, model="gpt-4o", temperature=0):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise

def extract_code_blocks(response):
    return re.findall(r"```python(.*?)```", response, re.DOTALL)

async def safe_execute(code_blocks, global_vars):
    from selenium.webdriver.chrome.service import Service

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(2),
        retry=tenacity.retry_if_exception_type((WebDriverException, TimeoutException)),
        reraise=True
    )
    def init_selenium():
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.binary_location = '/usr/bin/chromium'
        service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
        driver = webdriver.Chrome(service=service, options=options)
        logger.info("Selenium WebDriver initialized successfully.")
        return driver

    # Initialize dfs if not already present
    if 'dfs' not in global_vars:
        global_vars['dfs'] = {}
    
    for idx, code in enumerate(code_blocks):
        try:
            logger.info(f"Executing block {idx + 1}:\n{code.strip()}")
            if 'duckdb' in code:
                global_vars['con'] = initialize_duckdb()
            if 'webdriver' in code:
                driver = init_selenium()
                global_vars['webdriver'] = webdriver
                global_vars['Service'] = Service
                global_vars['ChromeDriverManager'] = ChromeDriverManager
                global_vars['ChromeType'] = ChromeType
                global_vars['By'] = By
                global_vars['WebDriverWait'] = WebDriverWait
                global_vars['EC'] = EC
                global_vars['driver'] = driver
                global_vars['default_timeout'] = 30
            if 'pdfplumber' in code:
                global_vars['pdfplumber'] = pdfplumber
            if 'tempfile' in code:
                global_vars['tempfile'] = tempfile
            if 'certifi' in code:
                global_vars['certifi'] = certifi
            if 'pytesseract' in code:
                global_vars['pytesseract'] = pytesseract
            if 'fuzz' in code:
                global_vars['fuzz'] = fuzz
            if 'clean_numeric_value' in code:
                global_vars['clean_numeric_value'] = clean_numeric_value
            if 'async' in code or 'await' in code:
                import asyncio
                async def run_async_code():
                    exec(code.strip(), global_vars)
                await asyncio.run(run_async_code())
            else:
                exec(code.strip(), global_vars)

            # Log dfs contents for debugging
            logger.info(f"global_vars['dfs'] keys: {list(global_vars.get('dfs', {}).keys())}")

            # Validate DataFrame storage
            has_valid_df = False
            if 'dfs' in global_vars and isinstance(global_vars['dfs'], dict) and global_vars['dfs']:
                for filename, df in global_vars['dfs'].items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        logger.info(f"Loaded DataFrame for {filename} with columns: {list(df.columns)}")
                        has_valid_df = True
                    else:
                        logger.warning(f"Invalid or empty DataFrame for {filename}: {type(df)}")
            if 'df' in global_vars and isinstance(global_vars['df'], pd.DataFrame) and not global_vars['df'].empty:
                logger.info(f"Loaded DataFrame with columns: {list(global_vars['df'].columns)}")
                global_vars['dfs']['default'] = global_vars['df']
                has_valid_df = True
            
            if not has_valid_df:
                logger.error("No valid non-empty DataFrame created.")
                return False, "No valid DataFrame created."
        except Exception as e:
            logger.error(f"Code block {idx + 1} failed: {e}")
            return False, str(e)
        finally:
            if 'driver' in global_vars and isinstance(global_vars['driver'], webdriver.Chrome):
                try:
                    global_vars['driver'].quit()
                    logger.info("Selenium driver closed.")
                except Exception as e:
                    logger.warning(f"Failed to close Selenium driver: {e}")
    return True, None
    
# Mapping of superscript digits to ASCII digits
SUPERSCRIPT_DIGIT_MAP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
}

def normalize_superscripts(value):
    """Convert superscript digits to ASCII equivalents and log superscript characters."""
    if not isinstance(value, str):
        value = str(value)
    
    # Log any superscript characters found
    superscript_chars = [char for char in value if ord(char) in range(0x2070, 0x209F) or ord(char) in [0x00B2, 0x00B3, 0x00B9]]
    if superscript_chars:
        logger.info(f"Superscript characters detected in value: '{value}' (chars: {superscript_chars})")
    
    # Normalize superscript digits
    for superscript, ascii in SUPERSCRIPT_DIGIT_MAP.items():
        value = value.replace(superscript, ascii)
    
    return value

def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        value = str(value).lower().strip()
        
        # Normalize superscript digits
        value = normalize_superscripts(value)
        logger.info(f"Normalized value: '{value}'")
        
        # Remove currency symbols, percentage, T, and any alphabetic prefix (including normalized superscripts)
        value = re.sub(r'^[a-zA-Z]+', '', value)  # Remove leading alphabetic characters
        value = re.sub(r'[\$₹€T]', '', value)     # Remove specific symbols
        
        # Handle percentage values
        if '%' in value:
            value = re.sub(r'[^\d.e-]', '', value)  # Remove all non-numeric except decimal, e, and minus
            return float(value) / 100  # Convert percentage to decimal
        elif 'billion' in value or 'bn' in value:
            value = float(re.sub(r'[^\d.e-]', '', value.replace('billion', '').replace('bn', ''))) * 1e9
        elif 'million' in value or 'mn' in value:
            value = float(re.sub(r'[^\d.e-]', '', value.replace('million', '').replace('mn', ''))) * 1e6
        elif 'crore' in value or 'cr' in value:
            value = float(re.sub(r'[^\d.e-]', '', value.replace('crore', '').replace('cr', ''))) * 1e7
        elif 'lakh' in value:
            value = float(re.sub(r'[^\d.e-]', '', value.replace('lakh', ''))) * 1e5
        else:
            value = float(re.sub(r'[^\d.e-]', '', value))
        return float(value)
    except (ValueError, TypeError):
        logger.error(f"Failed to clean value '{value}': returning np.nan")
        return np.nan
        

def infer_column_types(df):
    numeric_cols, categorical_cols, temporal_cols = [], [], []
    for col in df.columns:
        sample = df[col].dropna().head(10)
        if len(sample) < 2:
            categorical_cols.append(col)
            logger.warning(f"Column '{col}' has insufficient data; defaulting to categorical.")
            continue
        
        try:
            cleaned_sample = sample.apply(clean_numeric_value)
            numeric_sample = pd.to_numeric(sample, errors='coerce')
            if numeric_sample.notna().sum() >= len(sample) * 0.7:
                numeric_cols.append(col)
                continue
        except:
            pass
        
        try:
            temporal_sample = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
            if temporal_sample.notna().sum() >= len(sample) * 0.7:
                temporal_cols.append(col)
                continue
        except:
            pass
        
        categorical_cols.append(col)
    
    return numeric_cols, categorical_cols, temporal_cols

async def regenerate_with_error(messages, error_message, stage="step"):
    error_guidance = error_message
    if "HTTPConnectionPool" in error_message or "timeout" in error_message.lower():
        error_guidance += (
            "\nSelenium timed out while loading the page. Increase WebDriverWait timeout to 30 seconds and use EC.presence_of_element_located((By.TAG_NAME, 'table')). "
            "Add a timeout parameter to requests.get (e.g., timeout=30). Ensure network stability."
        )
    if "could not convert string to float" in error_message:
        error_guidance += (
            "\nCheck for non-numeric prefixes, suffixes, superscripts or annotations in numeric columns. "
            "Apply a cleaning function only to columns intended to be numeric based on question context. "
            "Remove superscripts."
            "Handle formats like '$1,234', '₹1,234', '1.2 billion', or '1.2 million' by scaling appropriately (e.g., to millions)."
            "Preserve categorical columns like 'Name', 'Symbol', or 'Company Name' without cleaning."
        )
    if "expected x and y to have same length" in error_message:
        error_guidance += (
            "\nEnsure regression and correlation calculations use only rows where both columns are non-null. "
            "Use df[['col1', 'col2']].dropna() to align data before passing to np.polyfit or corr()."
        )
    if "index out of bounds" in error_message or "no rows" in error_message:
        error_guidance += (
            "\nCheck for empty or insufficient data after filtering. Ensure DataFrame filtering returns non-empty results before accessing rows. "
            "Provide fallback values (e.g., 'None', 0.0) for edge cases."
        )
    if "No variable `result` or `results` found" in error_message:
        error_guidance += (
            "\nEnsure the final output is assigned to a variable named `result` (e.g., result = [...]). "
            "Do not only print the output; assign it explicitly to `result`."
        )
    if "print_png" in error_message and "optimize" in error_message:
        error_guidance += (
            "\nRemove the 'optimize' parameter from plt.savefig, as it is not supported. "
            "Use format='png' and adjust DPI (e.g., dpi=80) to ensure the base64 string is under 100,000 bytes."
        )
    if "no tables found" in error_message.lower():
        error_guidance += (
            "\nThe webpage may contain JavaScript-rendered content. "
            "Use Selenium with ChromeDriverManager(chrome_type=ChromeType.CHROMIUM) to match the installed Chromium version and render the page."
        )
    if "session not created" in error_message.lower() or "chromedriver" in error_message.lower():
        error_guidance += (
            "\nSelenium failed due to a ChromeDriver version mismatch. Use ChromeDriverManager(chrome_type=ChromeType.CHROMIUM) to automatically match the installed Chromium version (e.g., /usr/bin/chromium). "
            "Set options.binary_location='/usr/bin/chromium' in Selenium options."
        )
    if "ChromeDriverManager.__init__() got an unexpected keyword argument 'version'" in error_message.lower():
        error_guidance += (
            "\nRemove the 'version' parameter from ChromeDriverManager. Use ChromeDriverManager(chrome_type=ChromeType.CHROMIUM)."
        )
    if "no module named 'webdriver_manager.utils'" in error_message.lower():
        error_guidance += (
            "\nThe import 'webdriver_manager.utils' is incorrect. Use 'from webdriver_manager.core.os_manager import ChromeType' for webdriver-manager version 4.0.2."
        )
    if "unable to obtain driver for chrome" in error_message.lower():
        error_guidance += (
            "\nEnsure ChromeDriver is installed and accessible. Use Selenium with webdriver_manager.chrome.ChromeDriverManager to automatically install and manage ChromeDriver. "
            "Do not specify a manual path; let webdriver_manager handle it."
        )
    if "column" in error_message.lower() or "key" in error_message.lower():
        error_guidance += (
            "\nEnsure columns like 'Name', 'Symbol' (categorical), and 'Last Price', '% Change' (numeric) are preserved. "
            "Column name mismatch detected (e.g., 'Product Demand' vs. 'Product_Demand'). "
            "Do not assume specific column names like 'Name'. "
            "  Use relaxed fuzzy matching (e.g., fuzzywuzzy.fuzz.partial_ratio) to identify relevant columns. Accept matches with a similarity score as low as 50–60% "
            "Verify columns exist using df.columns before processing. Log available columns for debugging."
            "Do not apply numeric cleaning to categorical columns. Verify columns exist before processing."
        )
    if "julianday does not exist" in error_message.lower():
        error_guidance += (
            "\nThe DuckDB function 'julianday' is not available. Instead, load date columns into a pandas DataFrame and calculate date differences using "
            "pd.to_datetime() and .dt.days (e.g., (pd.to_datetime(df['date2']) - pd.to_datetime(df['date1'])).dt.days)."
        )
    if "failed to create directory" in error_message.lower() or "permission denied" in error_message.lower():
        error_guidance += (
            "\nDuckDB failed to create a directory due to permission issues. "
            "Use os.makedirs(os.getenv('DUCKDB_HOME', '/tmp/duckdb'), exist_ok=True) to create the DuckDB directory. "
            "Initialize DuckDB with an explicit database path: duckdb.connect(database=os.path.join(os.getenv('DUCKDB_HOME', '/tmp/duckdb'), 'duckdb.db')). "
            "Set the DuckDB temp_directory to os.getenv('DUCKDB_TEMP_DIR', '/tmp/duckdb') using con.execute('SET temp_directory=...')."
        )
    if "name 'model' is not defined" in error_message.lower():
        error_guidance += (
            "\nEnsure regression models (e.g., LinearRegression) are defined before use. "
            "Initialize the model variable (e.g., model = None) before any conditional blocks. "
            "In plotting code, check if the model exists before calling predict (e.g., if model is not None: plt.plot(...)). "
            "Use a fallback (e.g., flat line with plt.axhline) if the model cannot be computed due to insufficient data."
        )
    if "name 'Service' is not defined" in error_message.lower():
        error_guidance += (
            "\nThe 'Service' class from selenium.webdriver.chrome.service is missing. "
            "Ensure it is imported and added to global_vars in the safe_execute function."
        )
    if "no module named 'pdfplumber'" in error_message.lower():
        error_guidance += (
            "\nThe 'pdfplumber' module is missing. Ensure it is installed and imported correctly. "
            "Add 'import pdfplumber' to the code and include 'pdfplumber' in global_vars if used."
        )
    if "no module named 'pytesseract'" in error_message.lower():
        error_guidance += (
            "\nThe 'pytesseract' module is missing. Ensure it is installed and imported correctly. "
            "Add 'import pytesseract' to the code and include 'pytesseract' in global_vars if used."
        )
    if "Can only use .str accessor with string values" in error_message:
        error_guidance += (
            "\nThe .str accessor was used on a non-string column. "
            "Before applying .str methods, check column dtypes using df.dtypes and ensure the column is of object or string type. "
            "Use df[col].astype(str) only when necessary, and apply cleaning only to columns identified as numeric by infer_column_types. "
            "Log the column name and sample values to identify problematic data."
        )
    if "'int' object has no attribute 'lower'" in error_message.lower() or "object of type 'int' has no len()" in error_message.lower():
        error_guidance += (
            "\nFuzzy matching failed due to non-string column names (e.g., integers). Convert all column names to strings using `table.columns = table.columns.astype(str)` before fuzzy matching."
        )
    if "no relevant table found" in error_message.lower():
        error_guidance += (
            "\nNo table was selected due to overly restrictive fuzzy matching. "
            "Relax fuzzy matching to select a table if it contains at least one column matching "
            "Inspect all table columns and log them. Use fuzzy matching to map column names ."
        )
    if "no valid DataFrame created" in error_message.lower():
        error_guidance += (
            "\nNo valid DataFrame was stored in global_vars['dfs'] or global_vars['df']. "
            "Ensure the generated code stores DataFrames in `dfs` with filenames as keys (e.g., dfs['data.pdf'] = df_pdf) for multiple sources or in `df` and `dfs['default']` for a single source. "
            "For PDF files with multiple tables, concatenate tables into a single DataFrame using pd.concat(tables, ignore_index=True) if the question requires aggregated data. "
            "Log the contents of global_vars['dfs'] and global_vars['df'] after execution for debugging."
        )
    if "KeyError" in error_message.lower():
        error_guidance += (
            "\nA column name mismatch occurred . "
            "Use fuzzy matching (fuzzywuzzy.fuzz.partial_ratio) to map actual column names . "
            "Inspect DataFrame columns with df.columns.tolist() and log them. "
            "Select columns dynamically based on question context and available columns."
        )

    messages.append({
        "role": "user",
        "content": (
            f"The previous {stage} failed with this error:\n\n{error_guidance}\n\n"
            "Regenerate the {stage}. Inspect the DataFrame's columns, dtypes, and sample data (first 5 rows) and print them for debugging. "
            "Store processed DataFrames in a dictionary `dfs` with filenames as keys (e.g., dfs['data.pdf'] = df_pdf) for multiple sources or a single DataFrame in `df` and `dfs['default']` for a single source. "
            "For PDF files with multiple tables, concatenate tables into a single DataFrame using pd.concat(tables, ignore_index=True) if the question requires aggregated data (e.g., calculating averages across all tables). "
            "Do not assume specific column names. Use fuzzy matching (fuzzywuzzy.fuzz.partial_ratio) to select columns based on keywords from the question (e.g., 'name', 'company', 'symbol' for identifiers; 'change', 'percent' for metrics). "
            "Convert all column names to strings using `df.columns = df.columns.astype(str)` before fuzzy matching to handle non-string columns. "
            "Select columns based on question context and data types (numeric for metrics, categorical for identifiers, temporal for dates). "
            "Identify numeric, categorical, and temporal columns dynamically after cleaning data. "
            "Preserve categorical columns like 'Name', 'Symbol'. "
            "Clean numeric columns by removing non-numeric characters, prefixes, or annotations (e.g., 'T', 'RK'), and handling formats like '$1,234','Rs1,234', '1.2 billion', '1.2%', or '-0.13%'. "
            "Use StringIO for pd.read_html to avoid deprecation warnings. Drop rows with missing critical data for all required columns. "
            "Extract fields dynamically based on question context using flexible regular expressions and fuzzy matching (fuzzywuzzy.fuzz.partial_ratio). "
            "For web scraping, select the correct table by checking for relevant columns"
            "Use the correct import for ChromeType: `from webdriver_manager.core.os_manager import ChromeType` (do NOT use `webdriver_manager.core.utils`). "
            "For JavaScript-rendered content, use Selenium with ChromeDriverManager to handle WebDriver setup. "
            "For S3-based Parquet files, use DuckDB with hive_partitioning=True and limit queries to relevant subsets. "
            "For regressions or correlations, use only non-null data with df[['col1', 'col2']].dropna(). "
            "For plots, ensure base64 string is under 100,000 bytes by using format='png', figsize=(4,3), and dpi=80; reduce DPI further if needed."
            "Assign the final output to a variable named `result` (e.g., result = [...])."
        )
    })
    return await ask_gpt(messages)

async def process_question(question: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Data Analyst Agent tasked with answering arbitrary analysis questions by generating Python code. "
                "Assume the input data source is unknown (could be web, DuckDB queries, Parquet file reading, S3, local files, etc.). If the question specifies a URL, S3 path, or local file, generate code to fetch the data using libraries like pandas, requests, BeautifulSoup, or boto3. "
                "Break the task into clear steps, inspecting DataFrame columns to infer numeric, categorical, or temporal data dynamically after cleaning. "
                "When reading data from large remote sources (e.g., S3 Parquet paths or DuckDB queries over large partitions), avoid wildcard queries that may load excessive data. Instead, use lazy-loading patterns:"
                "- Restrict queries to specific subsets (e.g., one year if available) to avoid excessive data loading"
                "- Use DuckDB’s `read_parquet(..., hive_partitioning=True)` to avoid scanning all files unnecessarily."
                "- Use `df_iter = con.execute(query).fetchdf(stream=True)` for streaming where supported."
                "- Inspect available partitions first using `SELECT DISTINCT` on partition columns (`year`, `court`, etc.), then iterate selectively."
                "- Only expand the full query scope if needed by the question, and provide memory-efficient defaults during exploration."
                "For web scraping, fetch all tables from the specified URL using `requests` with `certifi` for SSL verification and `pandas.read_html` with `StringIO`. If no tables are found, fall back to Selenium with ChromeDriverManager to render JavaScript content. "
                "Inspect all tables and their column headings. Print the number of tables found and the column headings for each table to aid debugging. "
                "Convert all table column names to strings using `table.columns = table.columns.astype(str)` before fuzzy matching to handle non-string columns. "
                "Use fuzzy matching (fuzzywuzzy.fuzz.partial_ratio) to select the most relevant table and columns based on question context (e.g., 'name', 'company', 'symbol' for identifiers; 'change', 'percent' for metrics). "
                "If multiple tables match, select the one with the most relevant columns or the most rows.  "
                "If no table matches, log a warning and use a fallback (e.g., `selenium` for JavaScript-rendered content). "
                "Handle cases where tables are missing or dynamically loaded by suggesting fallback approaches (e.g., using `selenium` for JavaScript-rendered content)."
                "For PDF files, read using pdfplumber.open(file_path). Try extracting text with page.extract_text(). If no text is found, try OCR with pytesseract.image_to_string(page.to_image().original) for image-based PDFs, then extract tables with page.extract_tables(). "
                "For Excel files, read using pandas.read_excel(file_path). "
                "For CSV files, read using pandas.read_csv(file_path). "
                "For image files (e.g., PNG, JPG), use pytesseract.image_to_string(file_path) to extract text, then parse with regular expressions. "
                "Extract relevant data using regular expressions or table parsing based on the question’s context (e.g., company name, market cap, target price). "
                "In some cases like company financials, or company stock price, it can be in any currency."
                "Clean numeric columns by removing non-numeric characters, prefixes, or annotations (e.g., 'T', 'RK' in '24RK'). Handle formats like '$1,234', '1.2 billion', or '1.2 million' (scale appropriately). "
                "Clean numeric columns using `clean_numeric_value`, which normalizes superscript digits (e.g., '¹' to '1') and removes prefixes/suffixes (e.g., 'T', 'SM', 'ᴬ', 'ᴮ'). "                
                "Preserve categorical columns (e.g., 'title', 'name'). Convert temporal columns to datetime, handling various formats. Drop rows with missing critical data. "
                "Handle missing or malformed data by dropping rows or imputing sensibly based on the question’s requirements. "
                "For temporal columns, convert to datetime, handling formats like 'DD-MM-YYYY', 'YYYY-MM-DD', or others inferred from sample data."
                "Generate Python code only, executable locally, and store processed DataFrames in a dictionary `dfs` with filenames as keys (e.g., dfs['data.pdf'] = df_pdf) for multiple sources or a single DataFrame in `df` for a single source. "
                "Set `global_vars['dfs']` for multiple DataFrames or `global_vars['df']` for a single DataFrame. If a single DataFrame is created, also store it in `dfs['default']` for consistency."                "For questions requiring multiple outputs, format as a JSON array or object based on the question structure. "
                "For specific questions like 'scrape the list of highest-grossing films', return a JSON array of strings [int, string, float, base64 string] with raw values (e.g., '2', 'Titanic', '0.95', 'data:image/png;base64,...'), not formatted sentences. "
                "For plots, use matplotlib with figsize=(4,3), dpi=100, and encode to base64 using BytesIO, ensuring the base64 string is under 100,000 bytes (use format='png', reduce DPI if needed). "
                "Handle edge cases: return '0.0' for slopes or correlations if data is insufficient (e.g., <2 non-null rows); return 'None' for empty results in filtering operations. "
                "Assign the final output to a variable named `result` (e.g., result = [...]). Do not only print the output; ensure it is assigned to `result`."
                "Validate that the output matches the expected type and structure before returning."
            )
        }
    ]

    file_paths = {}
    if "Attachments:" in question:
        try:
            lines = [line.strip() for line in question.split('\n') if line.strip()]
            logger.info(f"Processed question lines: {lines}")
            attachment_start = -1
            for i, line in enumerate(lines):
                if line.startswith("Attachments:"):
                    attachment_start = i
                    break
            if attachment_start == -1:
                logger.error("No valid 'Attachments:' section found in question")
                return {"error": "No valid attachment section", "details": "Question contains 'Attachments:' but no valid section found"}
            
            attachment_details = lines[attachment_start + 1:]
            logger.info(f"Attachment details lines: {attachment_details}")
            if not attachment_details:
                logger.error("No file path provided in attachment details")
                return {"error": "No file path provided", "details": "Attachment details are empty"}
            
            # Parse multiple attachments
            for attachment_line in attachment_details:
                logger.info(f"Parsing attachment line: {attachment_line}")
                match = re.match(r"([^:]+):\s*(.+)", attachment_line)
                if not match:
                    logger.error(f"Invalid attachment format: {attachment_line}")
                    continue
                filename, path = match.groups()
                file_paths[filename.strip()] = path.strip()
                logger.info(f"Extracted file: {filename} -> {path}")
                if not os.path.exists(path.strip()):
                    logger.error(f"File not found at path: {path}")
                    file_paths[filename.strip()] = None  # Mark as invalid
        except Exception as e:
            logger.error(f"Failed to extract file paths from question: {e}")
            return {"error": "Failed to extract file paths", "details": str(e)}
    else:
        logger.info("No attachments specified; proceeding with question processing")
        
    messages.append({
        "role": "user",
        "content": (
            f"Analyze and break down this task into clear steps: {question}. "
            f"{'The question includes attachments with file paths: ' + str(file_paths) if file_paths else 'No attachments provided; assume the question may contain inline data or require external sources (e.g., web scraping).'} "
            "For multiple attachments, process each file based on its extension: use pdfplumber for .pdf, pandas.read_excel for .xlsx, pandas.read_csv for .csv, and pytesseract for images. "           
            "Identify the data source (e.g., URL, S3 path, local file) and fetch it appropriately. "
            "For S3-based Parquet files, inspect partitions with `SELECT DISTINCT` and limit queries to relevant subsets. "
            "For local PDF files, use a relative path (e.g., os.path.join(os.getcwd(), 'data', 'filename.pdf')). "
            "For remote PDF files, download using requests.get(url, stream=True, verify=certifi.where(), timeout=30) and save to a temporary file with tempfile.NamedTemporaryFile. "
            "For each step, describe how to inspect and handle data dynamically (e.g., inferring column types after cleaning, handling special prefixes like 'T'). "
            "If the question involves a specific URL, S3 path, or local file, include code to fetch the data in the first step, ensuring the correct table is selected by checking column names."
            "For web scraping, inspect all tables, print their column headings, and select the most relevant table based on the question’s context. "
            "If no tables are found, use Selenium with ChromeDriverManager to render the page and extract tables. "
            "Inspect all tables, print their column headings, and select the most relevant table based on the question’s context." 
            "Use fuzzy matching (fuzzywuzzy.fuzz.partial_ratio) to select columns based on question context (e.g., 'name', 'company', 'symbol' for identifiers; 'change', 'percent' for metrics). "                
            "Describe how to clean and analyze data dynamically, selecting columns based on context and data types."
        )
    })

    task_plan = await ask_gpt(messages)
    logger.info("Task Breakdown:\n" + task_plan)
    messages.append({"role": "assistant", "content": task_plan})

    global_vars = {"__name__": "__main__"}
    step_attempt = 0
    while step_attempt < MAX_ATTEMPTS:
        step_attempt += 1
                
        messages.append({
            "role": "user",
            "content": (
                "Write Python code to fetch and preprocess the data based on the task breakdown. "
                f"{'The question specifies attachments at paths: ' + str(file_paths) if file_paths else 'No attachments provided; check for inline data or external sources.'} "                "The question may specify processing files at URLs, S3 paths, local server paths, or temporary file paths (e.g., 'Attachments: filename: /tmp/...'). "
                "For each attachment, determine its type by extension and process accordingly:"                
                "- For PDFs, use pdfplumber.open(file_path) to extract text or tables; use pytesseract.image_to_string(page.to_image().original) for image-based PDFs. "
                "- For Excel files, use pandas.read_excel(file_path). "
                "- For CSV files, use pandas.read_csv(file_path). "
                "- For images (PNG, JPG), use pytesseract.image_to_string(file_path) to extract text, then parse with regular expressions. "
                "Verify each file is accessible before processing. If a file cannot be accessed, log an error and skip it. "
                "For remote files, download using requests.get(url, stream=True, verify=certifi.where(), timeout=30) and save to a temporary file with tempfile.NamedTemporaryFile."
                "For S3-based Parquet files, use DuckDB with `hive_partitioning=True`, inspect partitions with `SELECT DISTINCT`, and limit queries to relevant subsets. "
                "For web scraping, fetch all tables with `pandas.read_html` using `StringIO` and `requests`, with `certifi` for SSL verification, print column headings, and select the most relevant table. If no tables are found, use Selenium with ChromeDriverManager to render the page and extract tables.  "
                "Use the correct import: `from webdriver_manager.core.os_manager import ChromeType` (do NOT use `webdriver_manager.core.utils`). "
                "Extract data using regular expressions or table parsing to answer the question’s requirements."
                "Use fuzzy matching (fuzzywuzzy.fuzz.partial_ratio) to select columns based on question context (e.g., 'name', 'company', 'symbol' for identifiers; 'change', 'percent' for metrics). "
                "Convert all table column names to strings using `table.columns = table.columns.astype(str)` before fuzzy matching to handle non-string columns. "
                "Do not assume specific column names. Print DataFrame columns, dtypes, and sample data (first 5 rows) for debugging. "
                "Infer numeric, categorical, and temporal columns dynamically after cleaning data. "
                "Clean numeric columns by removing non-numeric characters, prefixes (e.g., 'T'), and handling formats like '$1,234' or '1.2 billion' (scale to millions). "
                "Clean numeric columns using `clean_numeric_value`, which normalizes superscript digits (e.g., '¹' to '1') and removes prefixes/suffixes (e.g., 'T', 'SM', 'ᴬ', 'ᴮ'). "
                "Log values containing superscript characters before and after cleaning. "
                "Use StringIO for pd.read_html to avoid deprecation warnings. Drop rows with missing critical data for all required columns. "
                "For web scraping, select the correct table by checking for relevant columns (e.g., 'Worldwide gross' instead of 'Gross')."
                "Convert temporal columns to datetime with flexible parsing. "
                "Drop rows with missing critical data for the question’s requirements."
                "Store processed DataFrames in a dictionary `dfs` with filenames as keys (e.g., dfs['data.pdf'] = df_pdf) for multiple sources or a single DataFrame in `df` for a single source (e.g., web scraping, single file). "
                "Set `global_vars['dfs']` for multiple DataFrames or `global_vars['df']` for a single DataFrame. If a single DataFrame is created, also store it in `dfs['default']` for consistency."
            )
        })
        code_response = await ask_gpt(messages)
        logger.info(f"[Attempt {step_attempt}] Step Code:\n{code_response}")

        code_blocks = extract_code_blocks(code_response)
        success, error = await safe_execute(code_blocks, global_vars)

        if success:
            break
        elif step_attempt < MAX_ATTEMPTS:
            code_response = await regenerate_with_error(messages, error, "step code")
            logger.info(f"[Regenerated Attempt {step_attempt + 1}] Step Code:\n{code_response}")
        else:
            return {"error": "Step code execution failed after max attempts", "details": error}

    metadata_info = "No dataframe created."
    if 'dfs' in global_vars and isinstance(global_vars['dfs'], dict) and global_vars['dfs']:
        metadata_info = ""
        for filename, df in global_vars['dfs'].items():
            if isinstance(df, pd.DataFrame):
                buffer = StringIO()
                df.info(buf=buffer)
                buffer.seek(0)
                metadata_info += f"\nFile: {filename}\n{buffer.getvalue()}"
                numeric_cols, categorical_cols, temporal_cols = infer_column_types(df)
                metadata_info += f"\nInferred Numeric Columns: {numeric_cols}"
                metadata_info += f"\nInferred Categorical Columns: {categorical_cols}"
                metadata_info += f"\nInferred Temporal Columns: {temporal_cols}"
                metadata_info += "\nSample data (first 5 rows):\n" + str(df.head(5))
            else:
                metadata_info += f"\nFile: {filename}\nNo valid DataFrame created."
    elif 'df' in global_vars and isinstance(global_vars['df'], pd.DataFrame):
        buffer = StringIO()
        global_vars['df'].info(buf=buffer)
        buffer.seek(0)
        metadata_info = f"\nSingle DataFrame:\n{buffer.getvalue()}"
        numeric_cols, categorical_cols, temporal_cols = infer_column_types(global_vars['df'])
        metadata_info += f"\nInferred Numeric Columns: {numeric_cols}"
        metadata_info += f"\nInferred Categorical Columns: {categorical_cols}"
        metadata_info += f"\nInferred Temporal Columns: {temporal_cols}"
        metadata_info += "\nSample data (first 5 rows):\n" + str(global_vars['df'].head(5))
        global_vars['dfs']['default'] = global_vars['df']
    logger.info(f"DataFrame Metadata:\n{metadata_info}")

    

    messages.append({
        "role": "user", 
        "content": (
            f"The dataframe metadata is:\n{metadata_info}\n\n"
            "Generate Python code to answer the question. Use the preprocessed DataFrames in `dfs`"
            "Determine which DataFrame to use based on the question context. "# (e.g., use dfs['data.pdf'] for questions about subjects and averages, dfs['exc.xlsx'] for questions about product demand). "
            "Use fuzzy matching (fuzzywuzzy.fuzz.partial_ratio) to select columns (e.g., 'name', 'company', 'symbol' for identifiers; 'change', 'percent' for metrics). "
            "Inspect columns and infer types (numeric, categorical, temporal) using `infer_column_types`. "
            "Select columns based on question context"
            "For temporal columns, convert to datetime, handling formats like 'DD-MM-YYYY', 'YYYY-MM-DD', or others inferred from sample data. "
            "Clean numeric columns by removing non-numeric characters, prefixes (e.g., 'T'), and handling formats like '$1,234' or '1.2 billion' (scale to millions). "
            "For regressions or correlations, use only non-null data with df[['col1', 'col2']].dropna(). "
            "For plots, use matplotlib with figsize=(4,3), dpi=100, and encode to base64 using BytesIO, ensuring the base64 string is under 100,000 bytes (use format='png', reduce DPI if needed). "
            "Handle edge cases: return '0.0' for slopes or correlations if data is insufficient (e.g., <2 non-null rows); return 'None' for empty results in filtering operations. "
            "The output format depends on the question: for questions like 'scrape the list of highest-grossing films', return a JSON array of strings [int, string, float, base64 string] with raw values (e.g., '2', 'Titanic', '0.95', 'data:image/png;base64,...'), not formatted sentences. "
            "Assign the final output to a variable named `result` (e.g., result = [...]). Do not only print the output; ensure it is assigned to `result`. "
            "Validate that the output matches the expected type and structure before returning."
        )
    })

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
            final_code = await regenerate_with_error(messages, error, "final code")
            logger.info(f"[Regenerated Attempt {final_attempt + 1}] Final Code:\n{final_code}")
        else:
            return {"error": "Final result code execution failed after max attempts", "details": error}

    try:
        result = global_vars.get("results", global_vars.get("result"))
        if result is None:
            raise ValueError("No variable `result` or `results` found.")

        # Attempt to repair and parse JSON strings
        if isinstance(result, str):
            try:
                result = repair_json(result)
                result = json.loads(result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse result as JSON: {e}")
                return {"error": "Result is a string but not valid JSON", "details": str(e)}

        # Validate that the result is JSON-serializable
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Result is not JSON-serializable: {e}")
            return {"error": "Result is not JSON-serializable", "details": str(e)}

        # Log the final result for debugging
        logger.info(f"Final result: {result}")
        return result

    except Exception as e:
        logger.error(f"Result extraction failed: {e}")
        return {"error": "Result extraction failed", "details": str(e)}

    

