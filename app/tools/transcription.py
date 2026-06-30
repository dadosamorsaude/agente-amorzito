from langchain_core.tools import tool
from langsmith import traceable
from app.core.config import settings
import logging
import os

logger = logging.getLogger(__name__)

# Pasta temporária para áudios (deve bater com a configuração do main.py)
UPLOAD_DIR = "temp_audios"

@tool
@traceable(name="transcribe_audio_tool")
async def transcribe_audio_tool(filename: str) -> str:
    """
    Transcreve um arquivo de áudio previamente enviado. 
    Use esta ferramenta quando o usuário enviar um áudio para análise.
    O 'filename' deve ser o nome do arquivo salvo no sistema.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        logger.info(f"Iniciando transcrição via MCP do arquivo: {file_path}")
        from app.services.mcp_client import invoke_mcp_tool
        
        response_obj = await invoke_mcp_tool(
            "transcribe_audio_tool",
            {"file_path": file_path, "agent_id": settings.AGENT_ID}
        )

        raw_text = ""
        if isinstance(response_obj, list):
            parts = []
            for item in response_obj:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            raw_text = "".join(parts)
        elif isinstance(response_obj, str):
            raw_text = response_obj
        else:
            raw_text = str(response_obj)

        return raw_text

    except Exception as e:
        logger.error(f"Erro na ferramenta de transcrição via MCP: {e}")
        return f"Erro ao transcrever o áudio via MCP: {str(e)}"
