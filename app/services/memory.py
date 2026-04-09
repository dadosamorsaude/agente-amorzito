from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_postgres import PostgresChatMessageHistory
import psycopg
import uuid
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# In-memory fallback (used when DATABASE_URL is not configured)
_memory_store: dict = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Returns a persistent PostgreSQL-backed chat history when DATABASE_URL is set,
    otherwise falls back to an in-memory store (not shared across workers).
    """
    # Ensure session_id is a valid UUID (required by langchain_postgres)
    # We use uuid5 to deterministically convert any string to a UUID
    try:
        uuid.UUID(session_id)
        valid_session_id = session_id
    except ValueError:
        # If not a valid UUID, generate a deterministic one from the string
        # Using a fixed namespace for consistency
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8') # DNS namespace
        valid_session_id = str(uuid.uuid5(namespace, session_id))

    if settings.DATABASE_URL:
        try:
            # Estabelece uma conexão síncrona com o PostgreSQL
            # Desabilitamos prepared statements por compatibilidade com o pooler do Supabase
            conn = psycopg.connect(settings.DATABASE_URL, prepare_threshold=None)
            
            # Garante que a tabela existe (necessário na nova implementação)
            PostgresChatMessageHistory.create_tables(conn, "chat_history")
            
            history = PostgresChatMessageHistory(
                "chat_history",
                valid_session_id,
                sync_connection=conn,
            )
            
            return history
        except Exception as e:
            logger.warning(
                f"Falha ao conectar ao PostgreSQL para memória, usando in-memory: {e}"
            )

    # Fallback: in-memory (single worker only)
    if session_id not in _memory_store:
        _memory_store[session_id] = ChatMessageHistory()
    return _memory_store[session_id]


def clear_session_history(session_id: str) -> None:
    """Clears conversation history for a given session."""
    # Deterministic UUID conversion
    try:
        uuid.UUID(session_id)
        valid_session_id = session_id
    except ValueError:
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        valid_session_id = str(uuid.uuid5(namespace, session_id))

    if settings.DATABASE_URL:
        try:
            conn = psycopg.connect(settings.DATABASE_URL, prepare_threshold=None)
            history = PostgresChatMessageHistory(
                "chat_history",
                valid_session_id,
                sync_connection=conn,
            )
            history.clear()
            return
        except Exception as e:
            logger.warning(f"Falha ao limpar histórico no PostgreSQL: {e}")

    if session_id in _memory_store:
        del _memory_store[session_id]
