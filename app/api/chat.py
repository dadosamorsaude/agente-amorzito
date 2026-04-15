import json
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.orchestrator import run_agent
from app.api.security import get_api_key
from app.core.logger import logger
from app.services.cache import semantic_cache
from fastapi import BackgroundTasks

router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str
    message: str
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    error: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Security(get_api_key),
):
    """
    Endpoint principal de chat.
    - stream=False: retorna JSON padrão
    - stream=True: retorna Server-Sent Events (SSE)
    """
    logger.info(f"Received chat request | user_id: {req.user_id} | stream: {req.stream}")

    # 1. Verifica no Cache Semântico
    cached_response = await semantic_cache.get(req.message)

    if req.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            # Se deu HIT no cache, entrega de uma vez no stream
            if cached_response:
                payload = {"text": cached_response}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
                
            try:
                full_stream_response = ""
                async for chunk in run_agent(req.user_id, req.message, stream=True):
                    if not chunk:
                        continue
                    
                    full_stream_response += chunk
                    payload = {"text": chunk}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"

                if full_stream_response and not full_stream_response.startswith("Erro técnico:"):
                    # Dispara salvamento em background para não travar a resposta
                    background_tasks.add_task(semantic_cache.set, req.message, full_stream_response)

            except Exception as e:
                logger.exception("Streaming error")
                payload = {"error": str(e)}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Tratamento quando stream=False
    try:
        # Se HIT no cache, não roda o run_agent
        if cached_response:
            return ChatResponse(
                response=cached_response,
                status="success",
                error=None,
            )

        full_response = ""

        async for chunk in run_agent(req.user_id, req.message, stream=False):
            if chunk:
                full_response += chunk

        if not full_response:
            return ChatResponse(
                response="",
                status="error",
                error="Nenhuma resposta foi gerada.",
            )

        if full_response.startswith("Erro técnico:"):
            return ChatResponse(
                response="",
                status="error",
                error=full_response,
            )
            
        background_tasks.add_task(semantic_cache.set, req.message, full_response)

        return ChatResponse(
            response=full_response,
            status="success",
            error=None,
        )

    except Exception as e:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=str(e))