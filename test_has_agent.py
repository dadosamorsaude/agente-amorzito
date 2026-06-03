import sys
import asyncio
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

sys.stdout.reconfigure(encoding='utf-8')

# Carregar variáveis de ambiente antes de importar os módulos do app
load_dotenv()

from app.agent.workers import clinical_has_tool

async def main():
    query = "Quais são os critérios para classificar um paciente hipertenso como risco alto e como eu faço a clusterização?"
    print(f"Testando o agente HAS com a query:\n'{query}'\n")
    
    config = RunnableConfig(configurable={"thread_id": "test_has_123"})
    
    try:
        # A ferramenta é assíncrona, então usamos ainvoke
        resultado = await clinical_has_tool.ainvoke({"query": query}, config=config)
        print("="*50)
        print("RESULTADO RETORNADO PELO AGENTE:")
        print("="*50)
        print(resultado)
        print("="*50)
    except Exception as e:
        print(f"Erro durante a execução: {e}")

if __name__ == "__main__":
    asyncio.run(main())
