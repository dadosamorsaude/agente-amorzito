from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from app.core.config import settings

def get_retriever(index_name: str):
    """
    Initializes and returns a specific Pinecone retriever.
    """
    if not settings.PINECONE_API_KEY:
        return None

    # O índice no Pinecone foi criado com dimensão 1024.
    # O modelo padrão do Langchain usa 1536, então precisamos forçar o uso da v3 com dimensions=1024.
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model="text-embedding-3-small",
        dimensions=1024
    )
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=settings.PINECONE_API_KEY
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def get_cfm_retriever():
    return get_retriever(settings.PINECONE_INDEX_CFM)

def get_pop_retriever():
    return get_retriever(settings.PINECONE_INDEX_POP)

@tool
def search_medical_compliance_tool(query: str) -> str:
    """
    OBRIGATÓRIO: Use esta ferramenta para buscar diretrizes CFM, Resolução 2.153/2016, 
    padrões de qualidade de prontuários, critérios de conformidade documental, 
    anamnese, conduta, hipótese diagnóstica, CID e boas práticas de registro clínico.
    """
    retriever = get_cfm_retriever()
    if not retriever: return "Erro ao configurar buscador CFM."
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs]) if docs else "Nenhuma diretriz encontrada."

@tool
def search_sop_tool(query: str) -> str:
    """
    OBRIGATÓRIO: Use esta ferramenta apenas para criação, revisão, estruturação 
    ou elaboração de POPs (Procedimento Operacional Padrão), arquitetura de POPs e modelos de procedimento.
    """
    retriever = get_pop_retriever()
    if not retriever: return "Erro ao configurar buscador de POPs."
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs]) if docs else "Nenhum POP encontrado."
