from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.core.config import settings


def get_chat_model_openai(
    temperature: float = None,
    model: str = None,
    run_name: str = None,
    metadata: dict = None,
):
    params = dict(
        model=model if model is not None else settings.MODEL_NAME,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )
    if run_name:
        params["run_name"] = run_name
    if metadata:
        params["metadata"] = metadata
    return ChatOpenAI(**params)


def get_chat_model_claude(
    temperature: float = None,
    run_name: str = None,
    metadata: dict = None,
):
    params = dict(
        model=settings.MODEL_CLAUDE,
        temperature=temperature if temperature is not None else settings.TEMPERATURE_CLAUDE,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    if run_name:
        params["run_name"] = run_name
    if metadata:
        params["metadata"] = metadata
    return ChatAnthropic(**params)