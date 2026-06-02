import logging
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from app.services.llm import get_chat_model_openai
from app.core.config import settings
from app.tools.athena import query_athena_tool
from app.tools.rag import search_medical_compliance_tool, search_sop_tool
from app.tools.transcription import transcribe_audio_tool
from app.tools.performance import analyze_clinical_performance_tool

logger = logging.getLogger(__name__)

# Agente Athena / Dados
@tool("athena_agent_tool")
async def athena_agent_tool(query: str, config: RunnableConfig) -> str:
    """
    Agente Especialista em Dados Clínicos e SQL (Athena).
    Use este agente SEMPRE que precisar consultar prontuários, pacientes, cid, anamnese, conduta, orientacao, etc no banco de dados.
    Passe as instruções claras sobre o que buscar e de quais datas.
    """
    logger.info("Executando Athena Agent...")
    llm = get_chat_model_openai(model=settings.MODEL_ATHENA)
    agent = create_react_agent(
        model=llm, 
        tools=[query_athena_tool],
        prompt="Você é um especialista em SQL para o AWS Athena. Retorne as informações pedidas."
    )
    child_config = {**config, "run_name": "Athena SQL Agent"}
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=child_config)
    return result["messages"][-1].content

# Agente de Compliance e Manuais
@tool("compliance_agent_tool")
async def compliance_agent_tool(query: str, config: RunnableConfig) -> str:
    """
    Agente Especialista em Compliance Médico, Regras CFM e POPs Internos.
    Use este agente para tirar dúvidas sobre como avaliar a qualidade, regras do CFM ou manuais operacionais.
    """
    logger.info("Executando Compliance Agent...")
    llm = get_chat_model_openai(model=settings.MODEL_COMPLIANCE)
    agent = create_react_agent(
        model=llm, 
        tools=[search_medical_compliance_tool, search_sop_tool],
        prompt="Você é um especialista em normas médicas e auditoria. Utilize suas ferramentas para buscar informações e repassá-las."
    )
    child_config = {**config, "run_name": "Compliance RAG Agent"}
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=child_config)
    return result["messages"][-1].content

# Agente de Áudio / Transcrição
@tool("audio_agent_tool")
async def audio_agent_tool(query: str, config: RunnableConfig) -> str:
    """
    Agente Especialista em Processamento de Áudio Clínico.
    Use este agente para transcrever ditados médicos.
    """
    logger.info("Executando Audio Agent...")
    llm = get_chat_model_openai(model=settings.MODEL_AUDIO)
    agent = create_react_agent(
        model=llm, 
        tools=[transcribe_audio_tool],
        prompt="Você transcreve e estrutura ditados médicos."
    )
    child_config = {**config, "run_name": "Audio Transcription Agent"}
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=child_config)
    return result["messages"][-1].content

# Agente de Performance e Auditoria
@tool("performance_agent_tool")
async def performance_agent_tool(query: str, config: RunnableConfig) -> str:
    """
    Agente Especialista em Relatórios de Performance Clínica e Qualidade Geral.
    Use este agente para obter taxas de conformidade, métricas gerais de auditoria e falhas comuns.
    """
    logger.info("Executando Performance Agent...")
    llm = get_chat_model_openai(model=settings.MODEL_PERFORMANCE)
    agent = create_react_agent(
        model=llm, 
        tools=[analyze_clinical_performance_tool],
        prompt="Você analisa relatórios de desempenho e métricas de auditoria."
    )
    child_config = {**config, "run_name": "Clinical Performance Agent"}
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=child_config)
    return result["messages"][-1].content
