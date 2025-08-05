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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=api_key)

MAX_ATTEMPTS = 4


# Ensure DuckDB uses the correct directory
def initialize_duckdb():
    duckdb_dir = os.getenv("DUCKDB_HOME", "/tmp/duckdb")
    try:
        logger.info(f"Initializing DuckDB with database path: {duckdb_dir}/duckdb.db")
        os.makedirs(duckdb_dir, exist_ok=True)
        con = duckdb.connect(database=f"{duckdb_dir}/duckdb.db")
        con.execute(f"SET temp_directory='{os.getenv('DUCKDB_TEMP_DIR', '/tmp/duckdb')}'")
        con.execute("SET http_timeout=30000")  # 30 seconds timeout
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

# Example usage in safe_execute or generated code
async def safe_execute(code_blocks, global_vars):
    from selenium.webdriver.chrome.service import Service  # Add this import
    for idx, code in enumerate(code_blocks):
        try:
            logger.info(f"Executing block {idx + 1}:\n{code.strip()}")
            # Ensure DuckDB is initialized before executing code
            if 'duckdb' in code:
                global_vars['con'] = initialize_duckdb()
            # Configure Selenium with compatible ChromeDriver
            if 'webdriver' in code:
                options = Options()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                # Use Chromium explicitly
                options.binary_location = '/usr/bin/chromium'
                # Auto-detect ChromeDriver version
                service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
                global_vars['webdriver'] = webdriver
                global_vars['Service'] = Service
                global_vars['ChromeDriverManager'] = ChromeDriverManager
                global_vars['ChromeType'] = ChromeType
                global_vars['By'] = By
                global_vars['WebDriverWait'] = WebDriverWait
                global_vars['EC'] = EC
                global_vars['options'] = options
                global_vars['service'] = service
                global_vars['default_timeout'] = 20
            # Handle async code (e.g., for future Playwright integration)
            if 'async' in code or 'await' in code:
                import asyncio
                async def run_async_code():
                    exec(code.strip(), global_vars)
                await asyncio.run(run_async_code())
            else:
                exec(code.strip(), global_vars)
            # Validate DataFrame
            if 'df' in global_vars and isinstance(global_vars['df'], pd.DataFrame):
                if global_vars['df'].empty:
                    logger.error("DataFrame is empty after loading.")
                    return False, "Empty DataFrame loaded."
                logger.info(f"Loaded DataFrame with columns: {list(global_vars['df'].columns)}")
            else:
                logger.error("No DataFrame created.")
                return False, "No DataFrame created."
        except Exception as e:
            logger.error(f"Code block {idx + 1} failed: {e}")
            return False, str(e)
        finally:
            # Clean up Selenium driver if created
            if 'driver' in global_vars and isinstance(global_vars['driver'], webdriver.Chrome):
                try:
                    global_vars['driver'].quit()
                    logger.info("Selenium driver closed.")
                except Exception as e:
                    logger.warning(f"Failed to close Selenium driver: {e}")
    return True, None

def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        value = str(value).lower().strip()
        value = re.sub(r'[\$₹€]', '', value)
        value = re.sub(r'^[a-zA-Z]+', '', value)
        #value = re.sub(r'[^0-9.e-]', '', value.replace('$', '').replace('₹', '').replace('t', '').replace('rk', '').replace('sm', '').replace('billion', 'e9').replace('million', 'e6').replace('\u1d40', '').replace('%', ''))
        #value = str(value).lower().replace(',', '').replace('$', '').replace('₹', '').replace('t', '')..replace('rk', '').replace('sm', '').replace('\u1d40', '').replace('%', '').strip()
        # Handle 'billion' or 'million' suffixes
        if 'billion' in value:
            value = float(re.sub(r'[^\d.e-]', '', value.replace('billion', ''))) * 1e9
        elif 'million' in value:
            value = float(re.sub(r'[^\d.e-]', '', value.replace('million', ''))) * 1e6
        else:
            # Remove non-numeric suffixes like 'RK'
            value = re.sub(r'[^\d.e-]', '', value)
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def infer_column_types(df):
    numeric_cols, categorical_cols, temporal_cols = [], [], []
    for col in df.columns:
        sample = df[col].dropna().head(5)
        if len(sample) == 0:
            categorical_cols.append(col)
            continue
        
        # Check for numeric content without cleaning
        try:
            numeric_sample = pd.to_numeric(sample, errors='coerce')
            if numeric_sample.notna().sum() >= len(sample) * 0.8:  # Allow 20% NaN tolerance
                numeric_cols.append(col)
                continue
        except:
            pass
        
        # Check for temporal content with flexible parsing
        try:
            temporal_sample = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
            if temporal_sample.notna().sum() >= len(sample) * 0.8:
                temporal_cols.append(col)
                continue
        except:
            pass
        
        # Analyze character composition for categorical columns
        sample_str = ' '.join(sample.astype(str))
        has_punctuation = any(c in sample_str for c in ['.', '_', '-', ':', '/'])
        alpha_count = sum(c.isalpha() for c in sample_str)
        digit_count = sum(c.isdigit() for c in sample_str)
        # Classify as categorical if it has punctuation or more alphabetic characters
        if has_punctuation or alpha_count >= digit_count:
            categorical_cols.append(col)
        else:
            categorical_cols.append(col)  # Default to categorical for safety
    
    return numeric_cols, categorical_cols, temporal_cols
    
"""
def infer_column_types(df, question_context=None):
    numeric_cols, categorical_cols, temporal_cols = [], [], []
    date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%Y']
    for col in df.columns:
        sample = df[col].dropna().head(5)
        if len(sample) == 0:
            categorical_cols.append(col)
            continue
        # Check character composition for alphabetic vs numeric
        sample_str = ' '.join(sample.astype(str))
        alpha_count = sum(c.isalpha() for c in sample_str)
        digit_count = sum(c.isdigit() for c in sample_str)
        # If more alphabetic characters than numeric, treat as categorical
        if alpha_count > digit_count:
            categorical_cols.append(col)
            continue
        # Try cleaning as numeric
        cleaned = sample.apply(clean_numeric_value)
        if cleaned.dropna().count() >= len(sample) * 0.7:
            numeric_cols.append(col)
            continue
        # Try as datetime
        is_temporal = False
        for date_format in date_formats:
            try:
                pd.to_datetime(sample, format=date_format, errors='raise')
                temporal_cols.append(col)
                is_temporal = True
                break
            except:
                continue
        if not is_temporal:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols, temporal_cols
"""
"""
def infer_column_types(df):
    numeric_cols, categorical_cols, temporal_cols = [], [], []
    date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d']
    for col in df.columns:
        sample = df[col].dropna().head(5)
        if len(sample) == 0:
            categorical_cols.append(col)
            continue
        # Try cleaning as numeric
        cleaned = sample.apply(clean_numeric_value)
        if cleaned.dropna().count() >= len(sample) * 0.5:
            numeric_cols.append(col)
            continue
        # Try as datetime
        try:
                pd.to_datetime(sample, format=date_format, errors='raise')
                temporal_cols.append(col)
                is_temporal = True
                break
        except:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols, temporal_cols
"""

async def regenerate_with_error(messages, error_message, stage="step"):
    error_guidance = error_message
    error_guidance = error_message
    if "HTTPConnectionPool" in error_message or "timeout" in error_message.lower():
        error_guidance += (
            "\nSelenium timed out while loading the page. Increase WebDriverWait timeout to 20 seconds and use EC.presence_of_element_located((By.TAG_NAME, 'table')). "
            "Add a timeout parameter to requests.get (e.g., timeout=30). Ensure network stability."
        )
    if "could not convert string to float" in error_message:
        error_guidance += (
            "\nCheck for non-numeric prefixes, suffixes, superscripts or annotations in numeric columns. "
            
            "Apply a cleaning function only to columns intended to be numeric based on question context. "
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
    # NEW: Handle missing result variable
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
            "\nThe 'version' parameter is not supported in this version of webdriver-manager. Use ChromeDriverManager(chrome_type=ChromeType.CHROMIUM) without specifying 'version' to auto-detect the installed Chromium version (138.0.7204.183). "
            "Alternatively, upgrade webdriver-manager to a version supporting explicit version specification."
        )
    if "no module named 'webdriver_manager.utils'" in error_message.lower():
        error_guidance += (
            "\nThe import 'webdriver_manager.utils' is incorrect. Use 'from webdriver_manager.core.os_manager import ChromeType' for webdriver-manager version 4.0.2."
        )
    if "unable to obtain driver for chrome" in error_message.lower():
        error_guidance += (
            "\nEnsure ChromeDriver is installed and accessible. Use webdriver_manager.chrome.ChromeDriverManager to automatically install and manage ChromeDriver. "
            "Do not specify a manual path; let webdriver_manager handle it."
        )
    if "column" in error_message.lower() or "key" in error_message.lower():
        error_guidance += (
            "\nEnsure columns like 'Name', 'Symbol' (categorical), and 'Last Price', '% Change' (numeric) are preserved. "
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

    messages.append({
        "role": "user",
        "content": (
            f"The previous {stage} failed with this error:\n\n{error_guidance}\n\n"
            "Regenerate the {stage}. Inspect the DataFrame's columns, dtypes, and sample data (first 5 rows) and print them for debugging. "
            "Select columns based on question context and data types (numeric for metrics, categorical for identifiers, temporal for dates). "
            "Identify numeric, categorical, and temporal columns dynamically after cleaning data. "
            "Preserve categorical columns like 'Name', 'Symbol'. "
            "Clean numeric columns by removing non-numeric characters, prefixes or superscripts or annotations (e.g., 'T', 'RK'), and handling formats like '$1,234' or '1.2 billion' (scale to millions). "
            "Use StringIO for pd.read_html to avoid deprecation warnings. Drop rows with missing critical data for all required columns. "
            "For web scraping, select the correct table by checking for relevant columns"
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
                # Data Source Handling
                "When reading data from large remote sources (e.g., S3 Parquet paths or DuckDB queries over large partitions), avoid wildcard queries that may load excessive data. Instead, use lazy-loading patterns:"
                "- Restrict queries to specific subsets (e.g., one year if available) to avoid excessive data loading"
                "- Add `LIMIT` clauses (e.g., `LIMIT 15000`) during exploration."
                "- Use DuckDB’s `read_parquet(..., hive_partitioning=True)` to avoid scanning all files unnecessarily."
                "- Use `df_iter = con.execute(query).fetchdf(stream=True)` for streaming where supported."
                "- Inspect available partitions first using `SELECT DISTINCT` on partition columns (`year`, `court`, etc.), then iterate selectively."
                "- Only expand the full query scope if needed by the question, and provide memory-efficient defaults during exploration."
                # Web Scraping and Table Selection
                "For web scraping, fetch all tables from the specified URL using `requests` with `certifi` for SSL verification and `pandas.read_html` with `StringIO`. If no tables are found, fall back to Selenium with ChromeDriverManager to render JavaScript content. "
                "Inspect all tables and their column headings. Print the number of tables found and the column headings for each table to aid debugging. "
                "Select the most relevant table based on the question’s context by matching column names to keywords relevant to the question (e.g., for a stock price question, prioritize columns like 'Price', 'Change', '% Change', 'Symbol', 'Name', or synonyms) or (e.g., for a film gross question, prioritize columns like 'Title', 'Worldwide gross', 'Year', 'Rank', 'Peak').  "
                "If exact column names are unknown, use the question’s context and data types (e.g., categorical for names, numeric for prices or changes) to select columns. "
                "If multiple tables match, select the one with the most relevant columns or the most rows.  "
                "If no table matches, log a warning and use a fallback (e.g., `selenium` for JavaScript-rendered content). "
                "Handle cases where tables are missing or dynamically loaded by suggesting fallback approaches (e.g., using `selenium` for JavaScript-rendered content)."
                # Data Cleaning
                "Clean numeric columns by removing non-numeric characters, prefixes, or annotations (e.g., 'T', 'RK' in '24RK'). Handle formats like '$1,234', '1.2 billion', or '1.2 million' (scale appropriately). "
                "Preserve categorical columns (e.g., 'title', 'name'). Convert temporal columns to datetime, handling various formats. Drop rows with missing critical data. "
                "Handle missing or malformed data by dropping rows or imputing sensibly based on the question’s requirements. "
                "For temporal columns, convert to datetime, handling formats like 'DD-MM-YYYY', 'YYYY-MM-DD', or others inferred from sample data."
                # Analysis and Output           
                "Generate Python code only, executable locally, and store the processed DataFrame in `df`. "
                "For questions requiring multiple outputs, format as a JSON array or object based on the question structure. "
                "For specific questions like 'scrape the list of highest-grossing films', return a JSON array of strings [int, string, float, base64 string] with raw values (e.g., '2', 'Titanic', '0.95', 'data:image/png;base64,...'), not formatted sentences. "
                "For plots, use matplotlib with figsize=(4,3), dpi=100, and encode to base64 using BytesIO, ensuring the base64 string is under 100,000 bytes (use format='png', reduce DPI if needed). "
                "Handle edge cases: return '0.0' for slopes or correlations if data is insufficient (e.g., <2 non-null rows); return 'None' for empty results in filtering operations. "
                # NEW: Enforce output variable naming
                "Assign the final output to a variable named `result` (e.g., result = [...]). Do not only print the output; ensure it is assigned to `result`."
                "Validate that the output matches the expected type and structure before returning."
            )
        },
        {
            "role": "user",
            "content": (
                f"Analyze and break down this task into clear steps: {question}. "
                "Identify the data source (e.g., URL, S3 path, local file) and fetch it appropriately. "
                "For S3-based Parquet files, inspect partitions with `SELECT DISTINCT` and limit queries to relevant subsets. "
                "For each step, describe how to inspect and handle data dynamically (e.g., inferring column types after cleaning, handling special prefixes like 'T'). "
                "If the question involves a specific URL, S3 path, or local file, include code to fetch the data in the first step, ensuring the correct table is selected by checking column names."
                "For web scraping, inspect all tables, print their column headings, and select the most relevant table based on the question’s context. "
                "If no tables are found, use Selenium with ChromeDriverManager to render the page and extract tables. "
                "Describe how to clean and analyze data dynamically, selecting columns based on context and data types."
            )
        }
    ]

    # Step 1: Task Breakdown
    task_plan = await ask_gpt(messages)
    logger.info("Task Breakdown:\n" + task_plan)
    messages.append({"role": "assistant", "content": task_plan})

    # Step 2: Step Code (includes data fetching if needed)
    global_vars = {"__name__": "__main__"}
    step_attempt = 0
    while step_attempt < MAX_ATTEMPTS:
        step_attempt += 1
        messages.append({
            "role": "user",
            "content": (
                "Write Python code to fetch and preprocess the data based on the task breakdown. "
                "Identify the data source from the question (e.g., S3 path, URL, local file). "
                "For S3-based Parquet files, use DuckDB with `hive_partitioning=True`, inspect partitions with `SELECT DISTINCT`, and limit queries to relevant subsets. "
                "For web scraping, fetch all tables with `pandas.read_html` using `StringIO` and `requests`, with `certifi` for SSL verification, print column headings, and select the most relevant table. If no tables are found, use Selenium with ChromeDriverManager to render the page and extract tables.  "
                "Do not assume specific column names. Print DataFrame columns, dtypes, and sample data (first 5 rows) for debugging. "
                "Infer numeric, categorical, and temporal columns dynamically after cleaning data. "
                "Clean numeric columns by removing non-numeric characters, prefixes (e.g., 'T'), and handling formats like '$1,234' or '1.2 billion' (scale to millions). "
                "Use StringIO for pd.read_html to avoid deprecation warnings. Drop rows with missing critical data for all required columns. "
                "For web scraping, select the correct table by checking for relevant columns (e.g., 'Worldwide gross' instead of 'Gross')."
                "Convert temporal columns to datetime with flexible parsing. "
                "Drop rows with missing critical data for the question’s requirements."
                "Store the DataFrame in `df` and set `global_vars['df']`."
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

    # Step 3: Metadata Summary
    metadata_info = "No dataframe created."
    if "df" in global_vars and isinstance(global_vars["df"], pd.DataFrame):
        try:
            df = global_vars["df"]
            buffer = StringIO()
            df.info(buf=buffer)
            buffer.seek(0)
            metadata_info = buffer.getvalue()
            numeric_cols, categorical_cols, temporal_cols = infer_column_types(df)
            metadata_info += f"\nInferred Numeric Columns: {numeric_cols}"
            metadata_info += f"\nInferred Categorical Columns: {categorical_cols}"
            metadata_info += f"\nInferred Temporal Columns: {temporal_cols}"
            metadata_info += "\nSample data (first 5 rows):\n" + str(df.head(5))
        except Exception as e:
            metadata_info = f"Error retrieving DataFrame metadata: {str(e)}"
    logger.info(f"DataFrame Metadata:\n{metadata_info}")

    # Step 4: Final Analysis Code
    messages.append({
        "role": "user", 
        "content": (
            f"The dataframe metadata is:\n{metadata_info}\n\n"
            "Generate Python code to answer the question. Use the preprocessed DataFrame `df`."
            "Inspect columns and infer types (numeric, categorical, temporal) using `infer_column_types`. "
            "Select columns based on question context"
            "For temporal columns, convert to datetime, handling formats like 'DD-MM-YYYY', 'YYYY-MM-DD', or others inferred from sample data. "
            "Clean numeric columns by removing non-numeric characters, prefixes (e.g., 'T'), and handling formats like '$1,234' or '1.2 billion' (scale to millions). "
            "For regressions or correlations, use only non-null data with df[['col1', 'col2']].dropna(). "
            "For plots, use matplotlib with figsize=(4,3), dpi=100, and encode to base64 using BytesIO, ensuring the base64 string is under 100,000 bytes (use format='png', reduce DPI if needed). "
            "Handle edge cases: return '0.0' for slopes or correlations if data is insufficient (e.g., <2 non-null rows); return 'None' for empty results in filtering operations. "
            "The output format depends on the question: for questions like 'scrape the list of highest-grossing films', return a JSON array of strings [int, string, float, base64 string] with raw values (e.g., '2', 'Titanic', '0.95', 'data:image/png;base64,...'), not formatted sentences. "
            # NEW: Enforce output variable naming
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

    # Step 5: Extract and Validate Result
    try:
        result = global_vars.get("results", global_vars.get("result"))
        if result is None:
            raise ValueError("No variable `result` or `results` found.")

        if isinstance(result, str):
            result = repair_json(result)
            result = json.loads(result)

        # Validate result format based on question structure
        if "json array" in question.lower() or "scrape the list of highest-grossing films" in question.lower():
            if not isinstance(result, list):
                raise ValueError("Expected JSON array output.")
            if "scrape the list of highest-grossing films" in question.lower():
                if not (len(result) == 4 and 
                        isinstance(result[0], str) and result[0].isdigit() and 
                        isinstance(result[1], str) and 
                        isinstance(result[2], str) and result[2].replace('.', '', 1).lstrip('-').isdigit() and 
                        isinstance(result[3], str) and result[3].startswith("data:image/png;base64,")):
                    raise ValueError("Expected JSON array of [int, string, float, base64 string] as strings.")
        elif "json object" in question.lower():
            if not isinstance(result, dict):
                raise ValueError("Expected JSON object output.")
        return result
    except Exception as e:
        return {"error": f"Result extraction failed: {e}"}

