import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.services.llm import get_chat_model_claude
from app.tools.athena import query_athena_tool
from app.tools.rag import search_medical_compliance_tool, search_sop_tool
from app.services.memory import get_session_history
from app.services.validator import validate_response
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
1. **CFM & Regulations**: For any questions regarding CFM (Conselho Federal de Medicina) guidelines, medical ethics, record-keeping standards (anamnese, conduta, etc.), and quality criteria, you MUST use the `search_medical_compliance_tool`.
2. **Standard Operating Procedures (POP)**: For questions about operational workflows, internal protocols, or creating/reviewing POPs, you MUST use the `search_sop_tool`.
3. **Internal Data**: Use `query_athena_tool` for patient data and specific medical records stored in the database.

## Database Schema (AWS Athena)
When querying medical records, use the following information:
- **Table**: `pdgt_amorsaude_inteligencia.tb_qualidade_prontuarios`
- **Allowed Columns**: id_agendamento, id_atendimento, data_atendimento, status_agendamento, id_procedimento, id_especialidade, especialidade, anamnese, conduta, hipotese_diagnostica, observacao, orientacao, solicitacao, especialidade_destino, cid_codigo, cid_descricao_detalhada, id_clinica, clinica, regional, uf, id_profissional, nome_profissional, prontuario_assinado.

## SQL Rules
- NEVER use `SELECT *`. List columns explicitly.
- Always filter by `data_atendimento` using the reference dates below.
- Limit detailed results to 20 rows.
- Use aggregations (COUNT, SUM, AVG) whenever possible for statistics.

## Date Reference
Today: {dates['hoje']}
Yesterday: {dates['ontem']}
"""


def normalize_content(content: Any) -> str:
    """
    Normalizes LangChain/LangGraph model content into plain text.
    Handles strings, lists of content blocks, dicts, and fallback values.
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
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                # Handles objects like content blocks or other types
                text_attr = getattr(item, "text", None)
                if text_attr is not None:
                    parts.append(str(text_attr))
                else:
                    parts.append(str(item))

        return "".join(parts)

    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return str(content)

    text_attr = getattr(content, "text", None)
    if text_attr is not None:
        return str(text_attr)

    return str(content)


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

    llm = get_chat_model_claude()
    tools = [
        query_athena_tool,
        search_medical_compliance_tool,
        search_sop_tool,
    ]
    dates = get_dates()
    system_prompt = _build_system_prompt(dates)
    history = get_session_history(user_id)

    try:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )

        input_messages = list(history.messages) + [HumanMessage(content=message)]

        config = {
            "configurable": {"thread_id": user_id},
            "run_name": "Agente Amorzito",
        }

        if stream:
            full_response = ""

            try:
                async for event in agent.astream_events(
                    {"messages": input_messages},
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event")

                    if kind == "on_tool_start":
                        tool_name = event.get("name", "ferramenta")
                        yield f"🔍 [Executando ferramenta: {tool_name}...]\n"
                        continue

                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if not chunk:
                            continue

                        text = normalize_content(getattr(chunk, "content", None))
                        if text:
                            full_response += text
                            yield text

                history.add_user_message(message)
                history.add_ai_message(full_response)

            except asyncio.CancelledError:
                logger.warning("Streaming cancelado pelo cliente.")
                return

        else:
            result = await agent.ainvoke({"messages": input_messages}, config=config)

            messages = result.get("messages", [])
            if not messages:
                final_response = "Não foi possível gerar uma resposta."
            else:
                response_text = normalize_content(messages[-1].content)
                validation = validate_response(response_text)
                final_response = validation.output

            history.add_user_message(message)
            history.add_ai_message(final_response)

            yield final_response

    except Exception as e:
        logger.exception("Erro no AgentExecutor")
        yield f"Erro técnico: {str(e)}"