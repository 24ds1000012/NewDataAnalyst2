import duckdb

def run_duckdb_analysis(s3_path: str, query: str):
    duckdb.sql("INSTALL httpfs; LOAD httpfs")
    duckdb.sql("INSTALL parquet; LOAD parquet")
    print (f"Running query on {s3_path} with DuckDB")
    return duckdb.sql(query).df()
