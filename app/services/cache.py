import hashlib
import json
import logging
import time
import uuid

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from app.core.config import settings

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 7
DEFAULT_THRESHOLD = 0.90
HIGH_THRESHOLD = 0.97
MIN_RESPONSE_LENGTH = 20

# Schema version: bump when tables/columns change in the system prompt
CACHE_SCHEMA_VERSION = hashlib.md5(
    "fl_qualidade_prontuarios_ia:"
    "id_paciente,data_nascimento,id_agendamento,id_atendimento,data_atendimento,"
    "status_agendamento,id_especialidade,especialidade,anamnese,conduta,"
    "hipotese_diagnostica,observacao,orientacao,solicitacao,especialidade_destino,"
    "cid_codigo,cid_descricao_detalhada,id_clinica,clinica,regional,uf,municipio,"
    "id_profissional,nome_profissional,prontuario_assinado".encode()
).hexdigest()


class PineconeSemanticCache:
    def __init__(self):
        self.enabled = bool(settings.PINECONE_API_KEY and getattr(settings, 'PINECONE_INDEX_CACHE', None))
        if self.enabled:
            try:
                self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                self.index = self.pc.Index(settings.PINECONE_INDEX_CACHE)
                self.embeddings = OpenAIEmbeddings(
                    api_key=settings.OPENAI_API_KEY,
                    model="text-embedding-3-small",
                    dimensions=1024
                )
            except Exception as e:
                logger.error(f"Erro ao inicializar PineconeSemanticCache: {e}")
                self.enabled = False

    async def get(self, query: str, user_id: str = "", threshold: float = DEFAULT_THRESHOLD):
        if not self.enabled:
            return None
        try:
            vector = await self.embeddings.aembed_query(query)
            result = self.index.query(vector=vector, top_k=3, include_metadata=True)

            if not result.matches:
                logger.info("Semantic Cache MISS")
                return None

            for match in result.matches:
                if match.score < threshold:
                    continue

                meta = match.metadata

                # Valida versão do schema
                if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
                    logger.info(f"Cache ignorado: versão do schema mudou | score={match.score:.4f}")
                    continue

                created_at = float(meta.get("created_at", 0))
                age_days = (time.time() - created_at) / 86400 if created_at else 0

                if age_days > CACHE_TTL_DAYS:
                    logger.info(f"Cache expirado (TTL) | score={match.score:.4f} | idade={age_days:.1f}d")
                    continue

                response = meta.get("response", "")
                if len(response.strip()) < MIN_RESPONSE_LENGTH:
                    logger.info(f"Cache ignorado: resposta muito curta | score={match.score:.4f}")
                    continue

                logger.info(f"Cache HIT | score={match.score:.4f} | idade={age_days:.1f}d")
                return meta

            logger.info("Semantic Cache MISS (nenhum match válido)")
            return None
        except Exception as e:
            logger.error(f"Erro no cache semantico (get): {e}")
            return None

    async def set(self, query: str, response: str, athena_data: list = None, rag_data: list = None, user_id: str = ""):
        if not self.enabled:
            return

        if len(response.strip()) < MIN_RESPONSE_LENGTH:
            logger.info(f"Cache: resposta muito curta, não armazenando")
            return

        if response.startswith("Erro técnico:") or response.startswith("Erro:"):
            logger.info(f"Cache: resposta de erro, não armazenando")
            return

        try:
            vector = await self.embeddings.aembed_query(query)
            metadata = {
                "query": query,
                "response": response,
                "athena_data": json.dumps(athena_data or []),
                "rag_data": json.dumps(rag_data or []),
                "user_id": user_id,
                "created_at": str(time.time()),
                "schema_version": CACHE_SCHEMA_VERSION,
            }
            self.index.upsert(vectors=[{
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": metadata,
            }])
            logger.info(f"Cache atualizado | user_id={user_id}")
        except Exception as e:
            logger.error(f"Erro no cache semantico (set): {e}")

    async def invalidate_by_score(self, query: str, user_id: str = ""):
        if not self.enabled:
            return
        try:
            vector = await self.embeddings.aembed_query(query)
            result = self.index.query(vector=vector, top_k=5, include_metadata=True)
            ids_to_delete = []
            for match in result.matches:
                if match.score > HIGH_THRESHOLD:
                    meta = match.metadata
                    if not user_id or meta.get("user_id") == user_id:
                        ids_to_delete.append(match.id)
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Cache invalidado: {len(ids_to_delete)} entrada(s) removida(s)")
        except Exception as e:
            logger.error(f"Erro ao invalidar cache: {e}")

    async def clear_all(self):
        if not self.enabled:
            return
        try:
            self.index.delete(delete_all=True)
            logger.info("Cache limpo completamente")
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")


semantic_cache = PineconeSemanticCache()
