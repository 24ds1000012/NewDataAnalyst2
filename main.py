from fastapi import FastAPI, UploadFile, File, HTTPException
#from fastapi.responses import Response
from fastapi.responses import JSONResponse
from agent import process_question
import uvicorn
import traceback
import json
from fastapi import Request
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)  # Create a logger instance

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Running"}

@app.post("/api/")

async def analyze(file: UploadFile = File(...)):
    try:
        question = (await file.read()).decode("utf-8")
        logger.info(f"Received question: {question}")
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        # Process question with timeout
        result = await asyncio.wait_for(process_question(question), timeout=300.0)
        logger.info(f"Returning result: {result}")
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


