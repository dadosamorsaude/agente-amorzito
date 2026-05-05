from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.agent.orchestrator import run_agent
from app.services.transcription import transcribe_audio
from app.core.logger import logger
import os
import uuid
import asyncio

router = APIRouter(tags=["websocket"])

UPLOAD_DIR = "temp_audios"

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket para Chat por Voz em Tempo Real (Protótipo).
    Recebe pedaços de áudio e processa ao finalizar.
    """
    await websocket.accept()
    user_id = str(uuid.uuid4()) # Em produção, receber do handshake
    logger.info(f"WebSocket conectado | session_id: {user_id}")

    temp_filename = f"ws_voice_{user_id}.mp3"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    try:
        with open(temp_path, "wb") as audio_file:
            while True:
                # Recebe dados do frontend
                data = await websocket.receive()
                
                if "bytes" in data:
                    # Se receber bytes, é pedaço de áudio
                    audio_file.write(data["bytes"])
                    # Feedback para o front que o áudio está chegando
                    # await websocket.send_json({"status": "recording", "msg": "Recebendo áudio..."})
                
                elif "text" in data:
                    # Se receber texto, pode ser um comando (ex: "FINISH")
                    msg = data["text"]
                    
                    if msg == "FINISH":
                        await websocket.send_json({"status": "processing", "msg": "Transcrevendo e analisando..."})
                        
                        # 1. Fecha o arquivo e transcreve
                        audio_file.close()
                        text = transcribe_audio(temp_path)
                        
                        if not text:
                            await websocket.send_json({"status": "error", "msg": "Não foi possível entender o áudio."})
                            continue

                        await websocket.send_json({"status": "transcribed", "text": text})

                        # 2. Chama o Agente (Streaming da resposta) com Auditoria Automática
                        full_query = (
                            f"Analise a seguinte transcrição de consulta médica e realize uma auditoria de conformidade "
                            f"baseada nas normas do CFM, RDCs e critérios de qualidade do AMORZITO:\n\n{text}"
                        )

                        full_response = ""
                        async for chunk in run_agent(user_id, full_query, stream=True):
                            if chunk:
                                full_response += chunk
                                await websocket.send_json({"status": "ai_reply", "chunk": chunk})
                        
                        await websocket.send_json({"status": "done", "full_response": full_response})
                        
                        # Reabre o arquivo para novas gravações se o socket continuar aberto
                        # (Opcional, dependendo da lógica do front)
                        # audio_file = open(temp_path, "ab")

                    elif msg == "PING":
                        await websocket.send_text("PONG")

    except WebSocketDisconnect:
        logger.info(f"WebSocket desconectado | session_id: {user_id}")
    except Exception as e:
        logger.error(f"Erro no WebSocket: {e}")
        await websocket.send_json({"status": "error", "msg": str(e)})
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
