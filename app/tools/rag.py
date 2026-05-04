from contextvars import ContextVar
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from app.core.config import settings

# Contexto por-task que armazena os trechos recuperados pelo RAG.
# Captura tanto CFM quanto POPs para uso pelo Agente Avaliador.
rag_results_context: ContextVar[list] = ContextVar("rag_results", default=[])

def get_retriever(index_name: str, namespace: str = ""):
    """
    Initializes and returns a specific Pinecone retriever.
    """
    if not settings.PINECONE_API_KEY:
        return None

    # 🔹 embeddings
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model="text-embedding-3-large",
        dimensions=3072
    )

    # 🔹 cliente Pinecone (OBRIGATÓRIO na versão nova)
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    index = pc.Index(index_name)

    # 🔹 vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})


def get_cfm_retriever(namespace: str = ""):
    return get_retriever(settings.PINECONE_INDEX_CFM, namespace=namespace)

def get_pop_retriever():
    return get_retriever(settings.PINECONE_INDEX_POP)

@tool
def search_medical_compliance_tool(query: str) -> str:
    """
    OBRIGATÓRIO: Use esta ferramenta para buscar diretrizes CFM, Resolução 2.153/2016, 
    padrões de qualidade de prontuários, critérios de conformidade documental, 
    anamnese, conduta, hipótese diagnóstica, CID e boas práticas de registro clínico.
    Também contém as regras de negócio do dashboard de qualidade.
    """
    # Busca em ambos os namespaces do novo index
    retriever_normas = get_cfm_retriever(namespace="cfm_2153_2016")
    retriever_regras = get_cfm_retriever(namespace="regras_negocio_prontuario")
    
    if not retriever_normas or not retriever_regras: 
        return "Erro ao configurar buscador de conformidade."
    
    docs_normas = retriever_normas.invoke(query)
    docs_regras = retriever_regras.invoke(query)
    
    all_docs = docs_normas + docs_regras
    
    # Captura os trechos recuperados no contexto para o Agente Avaliador
    captured = rag_results_context.get([])
    rag_results_context.set(captured + [{
        "source": "CFM/Regras",
        "query": query,
        "chunks": [d.page_content for d in all_docs]
    }])
    
    if not all_docs:
        return "Nenhuma diretriz ou regra encontrada."

    return "\n\n".join([d.page_content for d in all_docs])

@tool
def search_sop_tool(query: str) -> str:
    """
    OBRIGATÓRIO: Use esta ferramenta apenas para criação, revisão, estruturação 
    ou elaboração de POPs (Procedimento Operacional Padrão), arquitetura de POPs e modelos de procedimento.
    """
    retriever = get_pop_retriever()
    if not retriever: return "Erro ao configurar buscador de POPs."
    docs = retriever.invoke(query)

    # Captura os trechos recuperados no contexto para o Agente Avaliador
    captured = rag_results_context.get([])
    rag_results_context.set(captured + [{
        "source": "POP",
        "query": query,
        "chunks": [d.page_content for d in docs]
    }])

    return "\n\n".join([d.page_content for d in docs]) if docs else "Nenhum POP encontrado."
