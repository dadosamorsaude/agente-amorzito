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

        # Limita o contexto às últimas 10 conversas para economizar tokens lidos pelo LLM
        recent_messages = list(history.messages)[-10:]
        input_messages = recent_messages + [HumanMessage(content=message)]

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
                        logger.info(f"Executando ferramenta: {tool_name}")
                        continue

                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if not chunk:
                            continue

                        text = extract_text_from_content(getattr(chunk, "content", None))
                        if text:
                            full_response += text
                            yield text

                final_response = validate_response(full_response).output if full_response else ""

                history.add_user_message(message)
                history.add_ai_message(final_response)

            except asyncio.CancelledError:
                logger.warning("Streaming cancelado pelo cliente.")
                return

        else:
            result = await agent.ainvoke({"messages": input_messages}, config=config)

            messages = result.get("messages", [])
            if not messages:
                final_response = "Não foi possível gerar uma resposta."
            else:
                response_text = extract_text_from_content(messages[-1].content)
                validation = validate_response(response_text)
                final_response = validation.output

            history.add_user_message(message)
            history.add_ai_message(final_response)

            yield final_response

    except Exception as e:
        logger.exception("Erro no AgentExecutor")
        yield f"Erro técnico: {str(e)}"