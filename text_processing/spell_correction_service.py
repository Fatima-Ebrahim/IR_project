from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from text_processing.text_processing_handler import TextProcessingHandler

app = FastAPI()

text_handler = TextProcessingHandler()

class SpellCorrectionRequest(BaseModel):
    text: str

class SpellCorrectionResponse(BaseModel):
    original: str
    corrected: str

@app.post("/spell-correct", response_model=SpellCorrectionResponse)
def correct_spelling(request: SpellCorrectionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    corrected = text_handler.get_spell_corrected_text(request.text)
    return {
        "original": request.text,
        "corrected": corrected
    }