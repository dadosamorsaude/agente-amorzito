from typing import Optional
from langsmith import traceable
from app.skills.performance_audit import performance_audit_skill
from app.tools.athena import query_athena_tool
from app.core.logger import logger
import json

@traceable
async def analyze_clinical_performance_tool(limit: int = 10) -> str:
    """
    Ferramenta para análise de performance clínica em lote. 
    Busca os últimos prontuários no banco de dados e gera um relatório de qualidade e conformidade.
    Use esta ferramenta quando o usuário pedir um panorama geral, métricas de qualidade ou auditoria de múltiplos casos.
    """
    logger.info(f"Tool: Iniciando análise de performance clínica (limit={limit})")
    
    try:
        # 1. Busca os dados brutos no Athena usando a ferramenta existente
        # Ajuste o SQL conforme a estrutura da sua tabela
        query = f"SELECT conteudo_prontuario FROM atendimentos ORDER BY data_atendimento DESC LIMIT {limit}"
        athena_results_str = await query_athena_tool(query)
        
        # 2. Processa os resultados do Athena
        # O query_athena_tool retorna uma string (geralmente JSON ou formatada)
        # Vamos tentar extrair os textos dos prontuários
        try:
            records_data = json.loads(athena_results_str)
            # Assume que o resultado é uma lista de dicts com a chave 'conteudo_prontuario'
            records = [str(r.get('conteudo_prontuario', '')) for r in records_data if r.get('conteudo_prontuario')]
        except:
            # Fallback se não for JSON (caso o retorno do Athena seja formatado de outra forma)
            logger.warning("Não foi possível parsear retorno do Athena como JSON, tentando extração bruta.")
            records = [athena_results_str] # Usa o bloco inteiro como um registro único se falhar

        if not records or (len(records) == 1 and not records[0]):
            return "Não foram encontrados prontuários recentes no banco de dados para realizar a análise."

        # 3. Executa a Skill de Performance
        report = await performance_audit_skill.run_batch_audit(records)
        
        # 4. Retorna o relatório formatado para o Agente
        formatted_report = (
            f"### RELATÓRIO DE PERFORMANCE CLÍNICA (N={report['summary']['total_analyzed']})\n"
            f"- **Taxa de Conformidade**: {report['summary']['compliance_rate']}\n"
            f"- **Score Médio de Qualidade**: {report['summary']['average_quality_score']}/100\n\n"
            f"**Principais Falhas de Conformidade:**\n"
        )
        
        for issue in report['top_compliance_issues']:
            formatted_report += f"- {issue['item']} ({issue['occurrences']} ocorrências)\n"
            
        formatted_report += f"\n**Recomendação Gerencial:**\n{report['recommendation']}"
        
        return formatted_report

    except Exception as e:
        logger.error(f"Erro na ferramenta de análise de performance: {e}")
        return f"Erro ao realizar análise de performance: {str(e)}"
