import json

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app.chatbot.supervisor.shopping_supervisor import build_shopping_supervisor
from app.chatbot.supervisor.helpers.collect_product_ids import collect_product_ids
from app.schemas.chat_schema import ChatRequest

router = APIRouter(prefix="/api", tags=["chatbot"])


# Turn stored history into LangChain messages
def to_messages(request: ChatRequest) -> list:
    messages = [
        HumanMessage(content=item.content) if item.role == "user" else AIMessage(content=item.content)
        for item in request.history
    ]
    messages.append(HumanMessage(content=request.message))
    return messages


@router.post("/chat")
def chat(request: ChatRequest | None = None, questions: str | None = None):
    # Older callers send only ?questions=..., without history
    if request is None:
        if not questions:
            raise HTTPException(status_code=422, detail="message is required")
        request = ChatRequest(message=questions)

    try:
        shopping_supervisor = build_shopping_supervisor()
        result = shopping_supervisor.invoke({"messages": to_messages(request)})
        response = {"response": result["messages"][-1].content}

        # Add products only when the product search agent found some
        product_ids = collect_product_ids(result["messages"])
        if product_ids:
            response["products"] = product_ids
        print("response:", json.dumps(response, indent=2))

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
