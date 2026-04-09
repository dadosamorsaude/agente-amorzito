import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.services.llm import get_chat_model
from app.tools.athena import query_athena_tool
from app.services.rag import search_medical_compliance_tool, search_sop_tool
from app.services.memory import get_session_history
from app.services.validator import validate_response
from app.utils.dates import get_dates

logger = logging.getLogger(__name__)

# ── RAG detection keywords ──────────────────────────────────────────────────
RAG_KEYWORDS = [
    "qualidade", "prontuário", "avaliar", "avaliação", "cfm", "conformidade",
    "anamnese", "resolução", "conduta", "hipótese diagnóstica", "documentação",
    "registro clínico", "boas práticas", "conforme", "conformes", "pop",
    "criar pop", "produzir pop", "elabore um pop", "pops",
]

def _detect_rag(message: str) -> bool:
    """Returns True if the message requires RAG tools (CFM / POP)."""
    msg_lower = message.lower()
    return any(k in msg_lower for k in RAG_KEYWORDS)

def _build_system_prompt(dates: dict, rag_required: bool) -> str:
    """Builds the system prompt with dynamic date, RAG flag, and SQL schema injection."""
    return f"""You are AMORZITO, a medical record analysis assistant.
Always respond in Brazilian Portuguese.

## Objective
- Analyze medical records and quality/compliance indicators.
- Use RAG tools only when ragRequired = {rag_required}.

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

async def run_agent(user_id: str, message: str, stream: bool = False):
    """
    Main agent entrypoint. Yields tokens if stream=True, otherwise yields the full response once.
    """
    logger.info(f"Executando Agente Amorzito | user_id={user_id} | stream={stream}")

    if not message or not message.strip():
        yield "Por favor, digite uma mensagem."
        return

    rag_required = _detect_rag(message)
    llm = get_chat_model()
    tools = [query_athena_tool, search_medical_compliance_tool, search_sop_tool]
    dates = get_dates()
    system_prompt = _build_system_prompt(dates, rag_required)
    
    history = get_session_history(user_id)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(history.messages)
    messages.append(HumanMessage(content=message))

    try:
        agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

        if stream:
            full_response = ""
            async for event in agent.astream_events({"messages": messages}, version="v2"):
                kind = event["event"]
                
                # Signal that a tool is being called to keep the connection alive
                if kind == "on_tool_start":
                    tool_name = event["name"]
                    yield f"🔍 [Executando ferramenta: {tool_name}...]\n"
                
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        yield content
            
            # Persist history after stream end
            history.add_user_message(message)
            history.add_ai_message(full_response)
        else:
            # For non-streaming, we still yield the result as a single chunk
            result = await agent.ainvoke({"messages": messages})
            response_text = result["messages"][-1].content
            
            # Response validation
            validation = validate_response(response_text)
            final_response = validation.output
            
            history.add_user_message(message)
            history.add_ai_message(final_response)
            yield final_response

    except Exception as e:
        logger.exception("Erro no AgentExecutor")
        yield f"Erro técnico: {str(e)}"