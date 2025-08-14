from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from agent import process_question
import uvicorn
import traceback
import asyncio
import logging
import tempfile
import os
from starlette.datastructures import UploadFile as StarletteUploadFile

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
async def analyze(request: Request):
    # Initialize variables
    question_file = None
    attachments = []
    temp_file_paths = []

    # Log raw request body for debugging
    try:
        raw_body = await request.body()
        logger.info(f"Raw request body: {raw_body}")
    except Exception as e:
        logger.warning(f"Failed to read raw request body: {e}")

    # Try parsing form data (unlikely, but kept for compatibility)
    try:
        form = await request.form()
        logger.info(f"Form fields received: {dict(form.items())}")

        # Identify question file and attachments
        for field_name, field_value in form.items():
            if isinstance(field_value, StarletteUploadFile) and field_value.filename:
                fname = field_value.filename.lower()
                logger.info(f"Processing UploadFile: {field_name} -> {fname}")
                if fname.startswith("question") and fname.endswith(".txt"):
                    if question_file:
                        raise HTTPException(status_code=400, detail="Multiple question files provided")
                    question_file = field_value
                else:
                    attachments.append(field_value)
            else:
                logger.warning(f"Skipping non-file field: {field_name} = {field_value}")

        if not question_file:
            raise HTTPException(status_code=400, detail="No question file provided (expected 'question*.txt')")

        question_text = (await question_file.read()).decode("utf-8")
        if not question_text.strip():
            raise HTTPException(status_code=400, detail="Question file is empty")
        
        logger.info(f"Question text from file: {question_text}")

    except Exception as e:
        logger.warning(f"Form parsing failed: {e}. Attempting JSON body.")
        # Fallback to JSON body
        try:
            body = await request.json()
            logger.info(f"JSON body received: {body}")
            # Check for 'question', 'questions', or 'questions.txt' key
            question_key = None
            for key in ['question', 'questions', 'questions.txt', './questions.txt']:
                if key in body and isinstance(body[key], str) and (body[key].startswith('file://') or body[key].startswith('./')):
                    question_key = key
                    break
            if question_key:
                logger.info(f"Key: {question_key}")
                file_path = body[question_key].replace('file://', '')
                logger.info(f"Path: {file_path}")
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=400, detail=f"Question file not found locally: {file_path}. Ensure the file exists in the local directory.")
                with open(file_path, 'r', encoding='utf-8') as f:
                    question_text = f.read()
            else:
                raise HTTPException(status_code=400, detail="No 'question', 'questions', or 'questions.txt' provided in JSON body")
            
            if not question_text or not question_text.strip():
                raise HTTPException(status_code=400, detail="Question text is empty")

            # Handle attachments in JSON (any key with file:// except question-related keys)
            for key, value in body.items():
                if key in ['question', 'questions', 'questions.txt']:
                    continue
                if isinstance(value, str) and value.startswith('file://'):
                    file_path = value.replace('file://', '')
                    if not os.path.exists(file_path):
                        logger.warning(f"Attachment file not found locally: {file_path}. Skipping.")
                        continue
                    suffix = '.' + file_path.split('.')[-1] if '.' in file_path else ''
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        with open(file_path, 'rb') as f:
                            tmp_file.write(f.read())
                        temp_file_paths.append((key, tmp_file.name))
                        logger.info(f"Saved attachment: {key} -> {tmp_file.name}")
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.startswith('file://'):
                            file_path = item.replace('file://', '')
                            if not os.path.exists(file_path):
                                logger.warning(f"Attachment file not found locally: {file_path}. Skipping.")
                                continue
                            suffix = '.' + file_path.split('.')[-1] if '.' in file_path else ''
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                                with open(file_path, 'rb') as f:
                                    tmp_file.write(f.read())
                                temp_file_paths.append((file_path.split('/')[-1], tmp_file.name))
                                logger.info(f"Saved attachment: {file_path.split('/')[-1]} -> {tmp_file.name}")

        except Exception as e:
            logger.error(f"JSON parsing failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=400, detail="Invalid request: neither form data nor valid JSON provided. Ensure files exist locally.")

    # Save form attachments to temp files (for form data compatibility)
    for attachment in attachments:
        suffix = "." + attachment.filename.split(".")[-1] if "." in attachment.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await attachment.read()
            tmp_file.write(content)
            temp_file_paths.append((attachment.filename, tmp_file.name))
            logger.info(f"Saved attachment: {attachment.filename} -> {tmp_file.name}")

    # Append temp file paths to question string
    if temp_file_paths:
        question_text += "\nAttachments:\n" + "\n".join(f"{fname}: {path}" for fname, path in temp_file_paths)
    #logger.info(f"Final question text sent to process_question:\n{question_text}")

    # Process question
    try:
        result = await asyncio.wait_for(process_question(question_text), timeout=300.0)
    except Exception as e:
        logger.error(f"Error processing question: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

    # Cleanup temp files
    for _, path in temp_file_paths:
        try:
            os.unlink(path)
            logger.info(f"Cleaned up temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {path}: {e}")

    return JSONResponse(content=result)
