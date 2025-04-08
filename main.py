from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import logging
import time

app = FastAPI(title="Medical Imaging Analysis API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-imaging-backend")

# Request Body Model
class AnalysisRequest(BaseModel):
    image: str   # base64 image
    filename: str
    api_key: str

# Health Check
@app.get("/")
def root():
    return {"message": "Medical Imaging API is Running"}

# Analyze API
@app.post("/analyze")
async def analyze(request: AnalysisRequest):

    start_time = time.time()

    # Validate API key
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    try:
        client = OpenAI(api_key=request.api_key)

        # Call GPT-4o with image + prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI radiologist analyzing medical images. Provide a detailed assessment."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Perform a detailed radiological assessment of this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{request.image}"}}
                ]}
            ]
        )

        output = response.choices[0].message.content

        logger.info(f"Analysis completed for {request.filename}")

        end_time = time.time()

        return {
            "filename": request.filename,
            "analysis": output,
            "processing_time": round(end_time - start_time, 2)
        }

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
