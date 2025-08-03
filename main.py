from fastapi import FastAPI, UploadFile, File, HTTPException
#from fastapi.responses import Response
from fastapi.responses import JSONResponse
from agent import process_question
import uvicorn
import traceback
import json
from fastapi import Request

app = FastAPI()

@app.post("/api/")

async def analyze(file: UploadFile = File(...)):
    try:
        question = (await file.read()).decode("utf-8")
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        result = await process_question(question)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


"""
async def analyze(request: Request):
    try:
        content = await request.body()
        question = content.decode("utf-8")
        result = await process_question(question)

        #return Response(
        #    content=json.dumps(result),
        #    media_type="application/json"
        #)
        return JSONResponse(content=result)

    except Exception as e:
        return {
            "error": "Internal server error",
            "details": str(e),
            "traceback": traceback.format_exc(),
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
"""