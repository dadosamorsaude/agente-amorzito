from langchain_openai import ChatOpenAI
from app.core.config import settings

def get_chat_model(temperature: float = None):
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )