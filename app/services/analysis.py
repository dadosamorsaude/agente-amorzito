from langchain_core.prompts import ChatPromptTemplate
from app.services.llm import get_chat_model


ANALYSIS_SYSTEM_PROMPT = """
Você é AMORZITO, assistente de análise de prontuários.

Responda sempre em português do Brasil.

Objetivo:
- analisar qualidade e conformidade dos prontuários
- basear-se apenas nos dados fornecidos
- não alucinar
- usar linguagem cautelosa:
  "há indícios", "os registros sugerem", "pode haver oportunidade"

Se não houver dados suficientes, diga isso claramente.
"""


def analyze_data(message: str, data) -> str:
    llm = get_chat_model(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYSIS_SYSTEM_PROMPT),
            (
                "human",
                """
Pergunta original:
{message}

Dados retornados:
{data}

Gere uma resposta clara, objetiva e estruturada.
""",
            ),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({"message": message, "data": data})
    return result.content or ""