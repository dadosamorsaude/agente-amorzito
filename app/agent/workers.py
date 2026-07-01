import logging
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from app.services.llm import get_chat_model_openai
from app.core.config import settings
from app.services.mcp_client import query_athena_tool, search_medical_compliance_tool, search_sop_tool, search_clinic_has, get_cached_sql_expert_prompt
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
    
    # Busca prompt técnico do especialista SQL via MCP com cache
    try:
        sql_prompt = await get_cached_sql_expert_prompt("amorzito")
    except Exception as e:
        logger.warning(f"Erro ao carregar prompt SQL do MCP: {e}. Usando prompt de fallback local.")
        sql_prompt = (
            "You are an SQL specialist for AWS Athena. Return the requested information in a natural and clear way. "
            "NEVER expose or include the generated SQL query in your final response. ALWAYS respond in Brazilian Portuguese.\n"
            "## SQL Rules\n"
            "- NEVER use `SELECT *`. You must list the columns explicitly.\n"
            "- ALWAYS include `id_especialidade = 616` in the `WHERE` clause of every query to limit the scope.\n"
            "- Do NOT mix data types in `COALESCE`. `COALESCE(numeric_column, '')` causes TYPE_MISMATCH. Use `COALESCE(CAST(numeric_column AS VARCHAR), '')` or `COALESCE(numeric_column, 0)`.\n"
            "- When filtering dates, use `data_atendimento = DATE 'YYYY-MM-DD'` (or `>=` / `<` with DATE) directly to utilize partitions. Do NOT use `CAST(data_atendimento AS DATE)` as it breaks partition pruning.\n"
            "- **PUSH-DOWN vs NLP**: If analyzing THOUSANDS of patients, DO NOT select raw text columns (`anamnese`, `conduta`). Instead, use `CASE WHEN regexp_like(lower(col), 'pattern') THEN 1 ELSE 0 END` to classify them on the Athena server. HOWEVER, if analyzing a SMALL BATCH (e.g., LIMIT 50) for deep NLP evaluation, you MUST select the raw text columns so they can be read.\n"
            "- ALWAYS use `LIMIT 5000` or less when querying rows. Prioritize aggregations (`COUNT()`, `GROUP BY`) if you just need statistics.\n"
            "## Counting & Data Integrity Rules\n"
            "- When counting visits, consultations, or appointments, ALWAYS use `COUNT(DISTINCT id_atendimento)`.\n"
            "- When counting unique patients, ALWAYS use `COUNT(DISTINCT id_paciente)`.\n"
            "- Never use `COUNT(*)` generically, to avoid duplicate row inflations.\n"
            "- When returning individual records/lists, ALWAYS explicitly query and return `id_paciente`, `nome_paciente`, and `id_atendimento`. NEVER hallucinate, omit, or invent patient/appointment IDs.\n"
        )
        
    llm = get_chat_model_openai(model=settings.MODEL_ATHENA)
    agent = create_react_agent(
        model=llm, 
        tools=[query_athena_tool],
        prompt=sql_prompt
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
        prompt="You are an expert in medical norms and auditing. Use your tools to search for information and pass it on. ALWAYS respond in Brazilian Portuguese."
    )
    child_config = {**config, "run_name": "Compliance RAG Agent"}
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
        prompt="You analyze performance reports and auditing metrics. ALWAYS respond in Brazilian Portuguese."
    )
    child_config = {**config, "run_name": "Clinical Performance Agent"}
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=child_config)
    return result["messages"][-1].content

# Agente de Análise da Linha de Cuidado em Hipertensão Arterial
@tool("clinical_has_tool")
async def clinical_has_tool(query: str, config: RunnableConfig) -> str:
    """
    Agente Especialista em classificação de risco de pacientes em atendimento de cardiologia focado principalmente na Hipertensão Arterial.
    Use este agente para obter informações sobre a linha de cuidado em hipertensão arterial.
    Retorne as informações pedidas.
    """
    logger.info("Executando Avaliação Clínica HAS - HAS Agent...")
    llm = get_chat_model_openai(model=settings.MODEL_HAS)
    agent = create_react_agent(
        model=llm, 
        tools=[athena_agent_tool, search_clinic_has],
        prompt="You are an expert in Cardiology Appointments and risk classification of patients with Arterial Hypertension. Classify the risk of patients with HAS based on the documentation available via RAG. Return the requested information. ALWAYS respond in Brazilian Portuguese."
    )
    child_config = {**config, "run_name": "HAS Agent"}
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=child_config)
    return result["messages"][-1].content