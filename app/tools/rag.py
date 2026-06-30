from contextvars import ContextVar
from langchain_core.tools import tool
from langsmith import traceable
from app.core.config import settings

rag_results_context: ContextVar[list] = ContextVar("rag_results", default=[])


def format_docs(docs) -> str:
    """
    Formata os documentos recuperados com metadados.
    Ajuda o agente a saber de onde veio cada trecho.
    """
    if not docs:
        return ""

    formatted = []
    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        fonte = metadata.get("fonte", "Fonte não informada")
        artigo = metadata.get("artigo", "")
        tema = metadata.get("tema", "")
        capitulo = metadata.get("capitulo", "")
        secao = metadata.get("secao", "")

        header = (
            f"[Trecho {i}]\n"
            f"Fonte: {fonte}\n"
        )
        if capitulo:
            header += f"Capítulo: {capitulo}\n"
        if secao:
            header += f"Seção: {secao}\n"
        if artigo:
            header += f"Artigo: {artigo}\n"
        if tema:
            header += f"Tema: {tema}\n"

        formatted.append(
            f"{header}\nConteúdo:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


@tool
@traceable(name="search_medical_compliance_tool")
async def search_medical_compliance_tool(query: str) -> str:
    """
    OBRIGATÓRIO: Use esta ferramenta para buscar diretrizes CFM, Resolução CFM 2.153/2016,
    regras de negócio do dashboard de qualidade, critérios de conformidade documental,
    anamnese, conduta, hipótese diagnóstica, CID, prontuário assinado, IQRC,
    boas práticas de registro clínico e normas RDC/ANVISA relacionadas à qualidade,
    segurança do paciente, serviços de saúde, odontologia, resíduos, infraestrutura
    e processamento/esterilização.
    """
    try:
        from app.services.mcp_client import invoke_mcp_tool
        
        # CFM RAG search via MCP
        response_cfm = await invoke_mcp_tool(
            "search_rag_tool",
            {
                "query": query,
                "agent_id": settings.AGENT_ID,
                "namespace_key": "cfm",
                "k": 4
            }
        )
        
        # Rules RAG search via MCP
        response_regras = await invoke_mcp_tool(
            "search_rag_tool",
            {
                "query": query,
                "agent_id": settings.AGENT_ID,
                "namespace_key": "regras",
                "k": 4
            }
        )

        def parse_response(response_obj) -> tuple[str, list, list]:
            raw_text = ""
            if isinstance(response_obj, list):
                parts = []
                for item in response_obj:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    elif isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                    else:
                        parts.append(str(item))
                raw_text = "".join(parts)
            elif isinstance(response_obj, str):
                raw_text = response_obj
            else:
                raw_text = str(response_obj)
            return raw_text, [raw_text], []

        cfm_text, cfm_chunks, cfm_meta = parse_response(response_cfm)
        regras_text, regras_chunks, regras_meta = parse_response(response_regras)

        captured = rag_results_context.get([])
        rag_results_context.set(
            captured + [
                {
                    "source": "CFM",
                    "namespace": "cfm_2153_2016",
                    "query": query,
                    "chunks": cfm_chunks,
                    "metadata": cfm_meta,
                },
                {
                    "source": "Regras de Negócio",
                    "namespace": "regras_negocio_prontuario",
                    "query": query,
                    "chunks": regras_chunks,
                    "metadata": regras_meta,
                },
            ]
        )

        return f"=== DIRETRIZES CFM ===\n{cfm_text}\n\n=== REGRAS DE NEGÓCIO ===\n{regras_text}"

    except Exception as e:
        logger.error(f"Erro em search_medical_compliance_tool via MCP: {e}")
        return f"Erro ao acessar as diretrizes de compliance via MCP: {str(e)}."


@tool
@traceable(name="search_sop_tool")
async def search_sop_tool(query: str) -> str:
    """
    OBRIGATÓRIO: Use esta ferramenta apenas para criação, revisão, estruturação
    ou elaboração de POPs, Procedimento Operacional Padrão, arquitetura de POPs,
    modelos de procedimento, instruções operacionais e documentos internos de processo.
    """
    try:
        from app.services.mcp_client import invoke_mcp_tool
        
        # POP RAG search via MCP (namespace "rdc" covers POPs and RDCs based on config/agents.py)
        response_rdc = await invoke_mcp_tool(
            "search_rag_tool",
            {
                "query": query,
                "agent_id": settings.AGENT_ID,
                "namespace_key": "rdc",
                "k": 4
            }
        )

        raw_text = ""
        if isinstance(response_rdc, list):
            parts = []
            for item in response_rdc:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            raw_text = "".join(parts)
        elif isinstance(response_rdc, str):
            raw_text = response_rdc
        else:
            raw_text = str(response_rdc)

        captured = rag_results_context.get([])
        rag_results_context.set(
            captured + [
                {
                    "source": "RDC / POP (MCP)",
                    "query": query,
                    "chunks": [raw_text],
                    "metadata": [],
                }
            ]
        )

        return raw_text

    except Exception as e:
        logger.error(f"Erro em search_sop_tool via MCP: {e}")
        return f"Erro ao acessar POPs via MCP: {str(e)}."


@tool
@traceable(name="search_clinic_has")
async def search_clinic_has(query: str) -> str:
    """
    OBRIGATÓRIO: Use esta ferramenta para buscar informações sobre identificação, 
    clusterização, classificação de risco, e protocolo clínico de pacientes hipertensos (HAS).
    """
    try:
        from app.services.mcp_client import invoke_mcp_tool
        
        response_has = await invoke_mcp_tool(
            "search_rag_tool",
            {
                "query": query,
                "agent_id": settings.AGENT_ID,
                "namespace_key": "has",
                "k": 5
            }
        )

        raw_text = ""
        if isinstance(response_has, list):
            parts = []
            for item in response_has:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            raw_text = "".join(parts)
        elif isinstance(response_has, str):
            raw_text = response_has
        else:
            raw_text = str(response_has)

        captured = rag_results_context.get([])
        rag_results_context.set(
            captured + [
                {
                    "source": "Protocolo Clínico Hipertensão (MCP)",
                    "namespace": "documento_hipertensao",
                    "query": query,
                    "chunks": [raw_text],
                    "metadata": [],
                }
            ]
        )

        return raw_text

    except Exception as e:
        logger.error(f"Erro em search_clinic_has via MCP: {e}")
        return f"Erro ao consultar protocolo HAS via MCP: {str(e)}."