import json

from fastapi import APIRouter, HTTPException 
from app.chatbot.agents.shopping_agent import shopping_agent
router = APIRouter(prefix="/api", tags=["chatbot"]) 

@router.post("/chat")
def chat(questions: str):
    try: 
        response = shopping_agent(questions) 
        print("response:", json.dumps(response, indent=2))
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))