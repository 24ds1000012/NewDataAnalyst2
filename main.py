from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from agent import process_question
import uvicorn
import traceback
import json
import asyncio
import logging
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Running"}

@app.post("/api/")
async def analyze(
    questions_txt: UploadFile = File(..., alias="file"),
    attachments: list[UploadFile] = File(None, alias="attachments")
):
    temp_file_paths = []
    try:
        # Read the question file
        question = (await questions_txt.read()).decode("utf-8")
        logger.info(f"Received question: {question}")
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question file cannot be empty or contain only whitespace")

        # Collect all attachments from 'attachments' field and any other form fields
        # Parse form data manually
        form = await request.form()
        all_attachments = []
        if attachments:
            all_attachments.extend(attachments)
            logger.info(f"Received attachments via 'attachments' field: {[attachment.filename for attachment in attachments]}")

        # Check for additional form fields that contain UploadFile instances
        for field_name, field_value in extra_form_data.items():
            if isinstance(field_value, UploadFile):
                all_attachments.append(field_value)
                logger.info(f"Received attachment via '{field_name}' field: {field_value.filename}")
            elif isinstance(field_value, list) and all(isinstance(item, UploadFile) for item in field_value):
                all_attachments.extend(field_value)
                logger.info(f"Received attachments via '{field_name}' field: {[item.filename for item in field_value]}")

        if all_attachments:
            logger.info(f"Total attachments received: {[attachment.filename for attachment in all_attachments]}")
        else:
            logger.info("No attachments received")

        # Handle attachments
        for attachment in all_attachments:
            # Save each attachment to a temporary file
            suffix = f".{attachment.filename.split('.')[-1]}" if '.' in attachment.filename else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await attachment.read()
                if not content:
                    logger.warning(f"Attachment {attachment.filename} is empty")
                    continue
                tmp_file.write(content)
                temp_file_paths.append((attachment.filename, tmp_file.name))
                logger.info(f"Saved attachment {attachment.filename} to {tmp_file.name}")

        # Append temporary file paths to the question
        if temp_file_paths:
            question += "\nAttachments:\n" + "\n".join([f"{filename}: {path}" for filename, path in temp_file_paths])

        # Process the question
        logger.info(f"Total question plus attachments: {question}")
        result = await asyncio.wait_for(process_question(question), timeout=300.0)
        logger.info(f"Returning result: {result}")
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary files
        for _, temp_path in temp_file_paths:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.info(f"Deleted temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")