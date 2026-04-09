from fastapi import APIRouter, HTTPException, Security, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.agent.orchestrator import run_agent
from typing import Optional, AsyncGenerator
from app.core.logger import logger
from app.api.security import get_api_key
import json

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: str
    message: str
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    error: Optional[str] = None

@router.post("/chat")
async def chat(
    req: ChatRequest,
    api_key: str = Security(get_api_key)
):
    """
    Endpoint de chat principal com suporte a Streaming e Segurança via API Key.
    """
    logger.info(f"Received chat request | user_id: {req.user_id} | stream: {req.stream}")
    
    if req.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # O run_agent agora é sempre um gerador assíncrono
                async for token in run_agent(req.user_id, req.message, stream=True):
                    yield f"data: {json.dumps({'text': token})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    else:
        try:
            full_response = ""
            async for chunk in run_agent(req.user_id, req.message, stream=False):
                full_response += chunk
            return ChatResponse(response=full_response)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))