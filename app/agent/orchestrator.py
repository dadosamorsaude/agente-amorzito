import asyncio
import logging
import os
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from app.core.config import settings
from app.services.llm import get_chat_model_claude
from app.services.mcp_client import athena_results_context, rag_results_context
from app.agent.workers import (
    athena_agent_tool,
    compliance_agent_tool,
    performance_agent_tool
)
from app.services.memory import get_session_history
from app.utils.dates import get_dates

logger = logging.getLogger(__name__)


def _build_system_prompt(dates: dict) -> str:
    """Builds the system prompt with dynamic date and clear tool instructions."""
    return f"""You are AMORZITO, a medical record analysis assistant.
Always respond in Brazilian Portuguese.

## Objective
- Analyze medical records and quality/compliance indicators.
- Provide insights based on clinical data and official regulations.

## Guidelines for Quality & Compliance (RAG)
To ensure your responses are based on official evidence and protocols:
1. **CFM & Regulations / SOPs**: For any questions regarding guidelines, medical ethics, POPs, quality criteria, or **calculation of quality indicators (like IQRC)**, you MUST use the `compliance_agent_tool`.
2. **Internal Data**: Use `athena_agent_tool` for specific patient records, prescriptions, or direct database queries.
3. **Clinical Performance & Audit**: If asked about general quality, performance reports, or compliance trends, use `performance_agent_tool`.

## Diretrizes de Fidelidade Numérica e Integridade de Sessão
1. Fidelidade Numérica Absoluta: Transcreva os números gerados pelas consultas SQL exatamente como retornados. Nunca arredonde, estime ou modifique valores (por exemplo, se o SQL retornou 42, use '42', nunca 'cerca de 40').
2. Especificação da Métrica de Contagem: Sempre diferencie claramente o número de "atendimentos/consultas" e o número de "pacientes únicos" (por exemplo, 'X atendimentos referentes a Y pacientes únicos').
3. Menção de Período Temporal: Sempre informe claramente ao usuário qual o período de data_atendimento que foi considerado na contagem apresentada.
4. Identificadores Reais: Exiba apenas identificadores reais de pacientes (id_paciente e nome_paciente) e atendimentos (id_atendimento) conforme retornados pela consulta SQL. É expressamente proibido alucinar CPFs ou IDs fictícios.
5. Consistência de Filtros em Histórico: Ao processar perguntas consecutivas na mesma sessão, verifique o histórico para manter a consistência de data_atendimento, clínica, profissional ou outros filtros já aplicados, a menos que o usuário solicite explicitamente a alteração deles.

## Date Reference
Today: {dates['hoje']}
Yesterday: {dates['ontem']}
"""


def extract_text_from_content(content: Any) -> str:
    """
    Extracts only user-facing text from LangChain/LangGraph/Anthropic content blocks.
    Ignores tool_use, input_json_delta, and any non-text structured content.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                block_type = item.get("type")
                if block_type == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
                # ignora tool_use, input_json_delta e outros blocos
                continue

            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = getattr(item, "text", "")
                if text:
                    parts.append(str(text))

        return "".join(parts)

    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return ""

    content_type = getattr(content, "type", None)
    if content_type == "text":
        return str(getattr(content, "text", ""))

    return ""


async def _run_evaluation_background(
    user_id: str,
    message: str,
    final_response: str,
    raw_data: list,
    rag_data: list,
    chat_history: str = "",
) -> None:
    """Executa a avaliação em background."""
    try:
        logger.info(
            f"Avaliador iniciado em background | user_id={user_id} "
            f"| queries_athena={len(raw_data)} | queries_rag={len(rag_data)}"
        )
        evaluation = await evaluate_response(message, final_response, raw_data, rag_data, chat_history)
        await save_evaluation(user_id, message, final_response, raw_data, evaluation)
    except Exception:
        logger.exception("Erro no pipeline de avaliação em background")


async def run_agent(user_id: str, message: str, stream: bool = False):
    """
    Main agent entrypoint.
    - If stream=True, yields partial text chunks.
    - If stream=False, yields the final response once.
    """
    logger.info(f"Executando Agente Amorzito | user_id={user_id} | stream={stream}")

    if not message or not message.strip():
        yield "Por favor, digite uma mensagem."
        return

    # Zera os contextos de captura para esta execução
    athena_results_context.set([])
    rag_results_context.set([])

    env = os.getenv("RENDER", "development")
    tracing_metadata = {
        "user_id": user_id,
        "environment": env,
    }

    from app.services.llm import get_chat_model_openai

    primary_llm = get_chat_model_claude(model=settings.MODEL_ORCHESTRATOR, metadata=tracing_metadata)
    fallback_llm = get_chat_model_openai(model=settings.MODEL_NAME, metadata=tracing_metadata)
    llm = primary_llm.with_fallbacks([fallback_llm])
    tools = [
        athena_agent_tool,
        compliance_agent_tool,
        performance_agent_tool,
    ]
    dates = get_dates()
    
    # Carrega o prompt do sistema com cache TTL via MCP
    from app.services.mcp_client import get_cached_system_prompt
    try:
        system_prompt = await get_cached_system_prompt(
            agent_id="amorzito",
            data_hoje=dates['hoje'],
            data_ontem=dates['ontem']
        )
    except Exception as e:
        logger.warning(f"Erro ao carregar prompt com cache do MCP: {e}. Utilizando prompt de fallback local.")
        system_prompt = _build_system_prompt(dates)

    history = get_session_history(user_id)

    try:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )

        # Limita o contexto às últimas 10 conversas para economizar tokens lidos pelo LLM
        recent_messages = list(history.messages)[-10:]
        # OpenAI rejeita `name` em mensagens de role assistant/user/system.
        # Remove `name` de todas as mensagens do histórico (pode vir de versões anteriores
        # que usavam `create_react_agent(name="Agente Amorzito")`).
        for m in recent_messages:
            if hasattr(m, 'name'):
                m.name = None
            m.additional_kwargs.pop('name', None)
        input_messages = recent_messages + [HumanMessage(content=message)]

        config = {
            "configurable": {"thread_id": user_id},
            "run_name": "Agente Amorzito",
            "metadata": tracing_metadata,
            "tags": [env, "agent"],
        }
        
        TOOL_ALIASES = {
            "athena_agent_tool": "Pesquisa em Prontuários (SQL)",
            "compliance_agent_tool": "Análise de Conformidade",
            "performance_agent_tool": "Auditoria de Desempenho",
            "clinical_has_tool": "Classificação de Risco HAS",
            "query_athena_tool": "Consulta ao Banco de Dados Athena",
            "search_clinic_has": "Busca de Diretrizes (RAG)",
        }

        # Delega todo o processamento de eventos de streaming para o helper utilitário
        async for chunk in stream_agent_response(
            agent=agent,
            input_messages=input_messages,
            config=config,
            history=history,
            message=message,
            user_id=user_id,
            stream=stream,
            athena_results_context=athena_results_context,
            rag_results_context=rag_results_context,
            tool_aliases=TOOL_ALIASES
        ):
            yield chunk

    except Exception as e:
        logger.exception("Erro no AgentExecutor")
        yield f"Erro técnico: {str(e)}"