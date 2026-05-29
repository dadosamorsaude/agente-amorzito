import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from app.services.sql_generator import generate_sql
from app.agent.orchestrator import _build_system_prompt

def test_generate_sql_quality_table():
    """Verify that a quality-related question uses the correct system prompt with the new quality table."""
    message = "Quantos prontuários não foram assinados ontem em Belo Horizonte?"
    hoje = "2026-05-21"
    ontem = "2026-05-20"
    
    mock_response = AIMessage(
        content="SELECT COUNT(*) FROM pdgt_amorsaude_tecnologia.fl_qualidade_prontuarios_ia WHERE LOWER(regional) = 'bh' AND prontuario_assinado = 0 AND date(data_atendimento) >= DATE '2026-05-20' AND date(data_atendimento) < DATE '2026-05-21'"
    )
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_llm.return_value = mock_response  # In case it is called directly
    
    with patch("app.services.sql_generator.get_chat_model_openai", return_value=mock_llm) as mock_get_model:
        sql = generate_sql(message, hoje=hoje, ontem=ontem)
        
        # Verify the model was retrieved
        mock_get_model.assert_called_once()
        
        # Capture the prompt sent to the LLM (either via invoke or direct call)
        if mock_llm.invoke.called:
            called_args, called_kwargs = mock_llm.invoke.call_args
        else:
            called_args, called_kwargs = mock_llm.call_args
            
        prompt_value = called_args[0]
        
        # Convert PromptValue to messages
        messages = prompt_value.to_messages()
        system_msg = messages[0].content
        human_msg = messages[1].content
        
        # Assertions on the prompt content to verify it contains the updated rules and new table
        assert "pdgt_amorsaude_tecnologia.fl_qualidade_prontuarios_ia" in system_msg
        assert "tb_qualidade_prontuarios" not in system_msg
        # Assertions on human message formatting
        assert "2026-05-20" in human_msg
        assert "2026-05-21" in human_msg
        
        # Assertions on the output
        assert sql == mock_response.content

def test_orchestrator_system_prompt():
    """Verify that the orchestrator's system prompt includes the newly configured tables and schemas."""
    dates = {"hoje": "2026-05-21", "ontem": "2026-05-20"}
    prompt = _build_system_prompt(dates)
    
    assert "fl_qualidade_prontuarios_ia" in prompt
