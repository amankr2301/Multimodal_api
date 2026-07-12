import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Enable CORS (Crucial so the automated grader can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows requests from any website
    allow_credentials=True,
    allow_methods=["*"], # Allows all actions (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# 3. Define what incoming data must look like
class QA_Request(BaseModel):
    image_base64: str
    question: str

# 4. Connect to Gemini (Reads your API key from environment variables)
# We use gemini-2.5-flash as it is highly efficient at reading text from images
client = genai.Client()

@app.post("/answer-image")
async def answer_image(payload: QA_Request):
    try:
        # Step A: Clean and decode the base64 image data
        b64_data = payload.image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
            
        image_bytes = base64.b64decode(b64_data)
        
        # Step B: Strict instruction forcing Gemini to obey rule #1
        system_instruction = (
            "You are an expert data extraction bot. Answer the user's question using ONLY the provided image. "
            "CRITICAL: If the answer is a numeric value, output ONLY the number (e.g., '4089.35'). "
            "Do NOT include currency symbols ($ or ₹), units, spaces, or words. Just the raw value."
        )
        
        # Step C: Send the raw image bytes and question directly to Gemini
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png'
                ),
                payload.question
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0 # Low temperature makes the AI predictable and accurate
            )
        )
        
        # Step D: Return the response exactly as required by the spec
        final_answer = response.text.strip()
        return {"answer": final_answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

