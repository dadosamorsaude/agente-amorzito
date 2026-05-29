from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.core.config import settings


def _set_tracing(llm, run_name: str = None, metadata: dict = None):
    extra = {}
    if run_name:
        extra["run_name"] = run_name
    if metadata:
        extra["metadata"] = metadata
    if extra:
        llm.langsmith_extra = extra


def get_chat_model_openai(
    temperature: float = None,
    model: str = None,
    run_name: str = None,
    metadata: dict = None,
):
    llm = ChatOpenAI(
        model=model if model is not None else settings.MODEL_NAME,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )
    _set_tracing(llm, run_name, metadata)
    return llm


def get_chat_model_claude(
    temperature: float = None,
    run_name: str = None,
    metadata: dict = None,
):
    llm = ChatAnthropic(
        model=settings.MODEL_CLAUDE,
        temperature=temperature if temperature is not None else settings.TEMPERATURE_CLAUDE,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    _set_tracing(llm, run_name, metadata)
    return llm