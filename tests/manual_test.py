import asyncio
import sys
import os

# Adiciona o diretório raiz ao path para encontrar o módulo 'app'
sys.path.append(os.getcwd())

from app.agent.orchestrator import run_agent

async def main():
    user_id = "dev_test_user"
    print("\n🚀 AI Agent - Modo de Teste Manual")
    print("-----------------------------------")
    print("Comandos: 'exit' para sair, 'clear' para limpar contexto (pendente implementação)")
    
    while True:
        try:
            message = input("\n👤 Você: ")
            
            if message.lower() == 'exit':
                print("Encerrando...")
                break
            
            if not message.strip():
                continue

            print("🤖 Agente pensando...")
            response = await run_agent(user_id, message)
            
            print(f"\n✨ Resposta:\n{response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    asyncio.run(main())
