from fastapi import FastAPI
from app.api.recommend import router as recommend_router
from app.api.chatbot import router as chatbot_router
app = FastAPI()

app.include_router(recommend_router)
app.include_router(chatbot_router)