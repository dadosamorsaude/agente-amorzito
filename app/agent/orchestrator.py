import asyncio
import logging
from typing import Any
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from app.services.llm import get_chat_model_openai
from app.tools.athena import query_athena_tool, athena_results_context
from app.tools.rag import search_medical_compliance_tool, search_sop_tool, rag_results_context
from app.tools.transcription import transcribe_audio_tool
from app.tools.performance import analyze_clinical_performance_tool
from app.services.memory import get_session_history
from app.services.validator import validate_response
from app.utils.dates import get_dates
from app.agent.evaluator import evaluate_response
from app.services.evaluation_store import save_evaluation

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
1. **CFM & Regulations**: For any questions regarding CFM (Conselho Federal de Medicina) guidelines, medical ethics, record-keeping standards (anamnese, conduta, etc.), quality criteria, or **calculation of quality indicators (like IQRC)**, you MUST first use the `search_medical_compliance_tool` to validate current rules before querying the database.
2. **Standard Operating Procedures (POP)**: For questions about operational workflows, internal protocols, or creating/reviewing POPs, you MUST use the `search_sop_tool`.
3. **Internal Data**: Use `query_athena_tool` for specific patient records or direct database queries.
4. **Clinical Performance & Audit**: If asked about general quality, performance reports, or compliance trends, use `analyze_clinical_performance_tool`. Combine its quantitative metrics (compliance rate, common failures) with your detailed qualitative analysis.
5. **Audio Analysis (Auxiliar Médico)**: Use `transcribe_audio_tool` for clinical dictations. Always structure these as: ANAMNESE, CONDUTA, HIPÓTESE, CID-10 before auditing.

## Database Schema (AWS Athena)
When querying medical records, use the following information:
- **Table**: `pdgt_amorsaude_inteligencia.tb_qualidade_prontuarios`
- **Allowed Columns**: id_agendamento, id_atendimento, data_atendimento, status_agendamento, id_procedimento, id_especialidade, especialidade, anamnese, conduta, hipotese_diagnostica, observacao, orientacao, solicitacao, especialidade_destino, cid_codigo, cid_descricao_detalhada, id_clinica, clinica, regional, uf, id_profissional, nome_profissional, prontuario_assinado.

## SQL & Analysis Rules
- **Fields to Analyze**: Always focus on `anamnese`, `conduta`, `hipotese_diagnostica`, `cid_codigo` and `prontuario_assinado`.
- **Quality Logic (IQRC)**: A record is only considered compliant (IQRC success) if `anamnese`, `conduta`, `hipotese_diagnostica`, `cid_codigo`, AND `prontuario_assinado` are all valid/signed.
- **Text Validation**: Fields filled with "xxx", "--", "ok", "NA", ".....", or generic text are considered **NOT filled**.
- **Signed Status**: A record is signed only if `prontuario_assinado` is 'Sim'.
- **Valid Appointments**: Only consider records where `status_agendamento` is one of: 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 24, 40, 60, 83.
- **Mandatory Exclusions**: ALWAYS exclude `id_especialidade` IN (932, 1154, 993, 776, 777, 892, 1013, 711, 778, 658, 712, 732, 680, 1274, 779).
- **SQL Best Practices**: NEVER use `SELECT *`. List columns explicitly.
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


async def _run_evaluation_background(
    user_id: str,
    message: str,
    final_response: str,
    raw_data: list,
    rag_data: list,
    chat_history: str = "",
) -> None:
    """Executa a avaliação em background sem bloquear a resposta ao usuário."""
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

    llm = get_chat_model_openai()
    tools = [
        query_athena_tool,
        search_medical_compliance_tool,
        search_sop_tool,
        transcribe_audio_tool,
        analyze_clinical_performance_tool,
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

                # Dispara avaliação em background (sem bloquear o stream)
                raw_data = athena_results_context.get([])
                rag_data = rag_results_context.get([])
                if (raw_data or rag_data) and final_response:
                    # Formata o histórico para o avaliador
                    history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in recent_messages])
                    asyncio.create_task(
                        _run_evaluation_background(user_id, message, final_response, raw_data, rag_data, history_str)
                    )

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

            # Dispara avaliação em background (sem bloquear a resposta)
            raw_data = athena_results_context.get([])
            rag_data = rag_results_context.get([])
            if (raw_data or rag_data) and final_response:
                # Formata o histórico para o avaliador
                history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in recent_messages])
                asyncio.create_task(
                    _run_evaluation_background(user_id, message, final_response, raw_data, rag_data, history_str)
                )

            yield final_response

    except Exception as e:
        logger.exception("Erro no AgentExecutor")
        yield f"Erro técnico: {str(e)}"