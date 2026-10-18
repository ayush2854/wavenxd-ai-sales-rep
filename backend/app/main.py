"""FastAPI app. Run from the backend folder:  uvicorn app.main:app --reload"""
from __future__ import annotations

import os
import uuid
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import SalesAgent
from .prompts import ASSISTANT_NAME

load_dotenv()

app = FastAPI(title="WaveNxD AI Sales Representative")

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_agent() -> SalesAgent:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return SalesAgent()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "assistant": ASSISTANT_NAME}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, agent: SalesAgent = Depends(get_agent)) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")
    session_id = req.session_id or str(uuid.uuid4())
    return ChatResponse(reply=agent.reply(session_id, message), session_id=session_id)


@app.post("/reset")
def reset(session_id: str, agent: SalesAgent = Depends(get_agent)) -> dict[str, str]:
    agent.reset(session_id)
    return {"status": "reset"}