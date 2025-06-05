from fastapi import APIRouter, Request
from api.services.summarizer_service import summarize_text

router = APIRouter()

@router.post("/summarize")
async def summarize(request: Request):
    body = await request.json()
    text = body.get("text")
    if not text:
        return {"error": "No text provided"}
    
    summary = summarize_text(text)
    return {"summary": summary}
