from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .spell_correction_handler import SpellCorrectionHandler

app = FastAPI()
spell_corrector = SpellCorrectionHandler()

class SpellCorrectionRequest(BaseModel):
    text: str

class SpellCorrectionResponse(BaseModel):
    original: str
    corrected: str

@app.post("/spell-correct", response_model=SpellCorrectionResponse)
def correct_spelling(request: SpellCorrectionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    corrected = spell_corrector.correct_spelling(request.text)
    return {
        "original": request.text,
        "corrected": corrected
    }