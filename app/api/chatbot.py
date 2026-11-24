import json

from fastapi import APIRouter, HTTPException
from app.chatbot.supervisor.shopping_supervisor import run_shopping_supervisor
router = APIRouter(prefix="/api", tags=["chatbot"])

@router.post("/chat")
def chat(questions: str):
    try:
        response = run_shopping_supervisor(questions)
        print("response:", json.dumps(response, indent=2))
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
