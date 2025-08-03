---
title: NewDataAnalyst
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.27.0"
app_file: app.py
pinned: false
---

# Data Analyst Agent API

This is a FastAPI-based agent that uses GPT to process arbitrary data analysis questions. It supports web scraping, DuckDB over Parquet, regression analysis, and plotting.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload

## API Endpoint

**POST** `/api/`  
Upload a `.txt` file containing the analysis question.  
Returns a JSON response with 4 elements: `[score (float), topic (str), correlation (float), image (base64 str)]`.
