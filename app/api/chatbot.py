import json

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.chatbot.supervisor.shopping_supervisor import build_shopping_supervisor
from app.chatbot.supervisor.tools.delegate_to_product_search_agent import collect_product_ids

router = APIRouter(prefix="/api", tags=["chatbot"])

@router.post("/chat")
def chat(questions: str):
    try:
        shopping_supervisor = build_shopping_supervisor()
        result = shopping_supervisor.invoke({"messages": [HumanMessage(content=questions)]})

        response = {"response": result["messages"][-1].content}

        # Add products only when the product search agent found some
        product_ids = collect_product_ids(result["messages"])
        if product_ids:
            response["products"] = product_ids

        print("response:", json.dumps(response, indent=2))
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
