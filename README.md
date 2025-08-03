---
title: Data Analyst Agent
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "3.10"
app_file: main.py
pinned: false
---
Data Analyst Agent API
This is a FastAPI-based agent that uses GPT to process arbitrary data analysis questions. It supports web scraping, DuckDB over Parquet, regression analysis, and plotting.

**Run Locally**
```bash
pip install -r requirements.txt
uvicorn main:app --reload