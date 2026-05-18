from langchain_core.prompts import ChatPromptTemplate
from app.services.llm import get_chat_model_openai


ANALYSIS_SYSTEM_PROMPT = """
Você é AMORZITO, assistente de análise de prontuários.

Responda sempre em português do Brasil.

Objetivo:
- analisar qualidade e conformidade dos prontuários (foco em: anamnese, conduta, hipótese, cid e assinatura)
- considerar campos com "xxx", "--", "ok", "NA" ou textos genéricos como **NÃO PREENCHIDOS**
- o IQRC (indicador principal) exige preenchimento simultâneo de todos os campos: anamnese, conduta, hipótese, cid e assinatura
- basear-se apenas nos dados fornecidos
- não alucinar
- usar linguagem cautelosa:
  "há indícios", "os registros sugerem", "pode haver oportunidade"

Se não houver dados suficientes, diga isso claramente.
"""


def analyze_data(message: str, data) -> str:
    llm = get_chat_model_openai(temperature=0.2)

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