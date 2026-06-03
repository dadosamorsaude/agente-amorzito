import os
import json
from pathlib import Path
from dotenv import load_dotenv
import docx
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "rag-agente-hipertensao"
NAMESPACE = "documento_hipertensao"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072

FILEPATH = r"rag\Identificação e Clusterização de Pacientes Hipertensos por Grau de Risco.docx"
OUTPUT_JSONL = "rag/chunks_hipertensao.jsonl"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", ";", " "]
)

def extract_text_from_docx(filepath: str) -> str:
    doc = docx.Document(filepath)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def build_chunks(filepath: str) -> list[dict]:
    raw_text = extract_text_from_docx(filepath)
    sub_chunks = splitter.split_text(raw_text)
    chunks = []
    for i, sub in enumerate(sub_chunks, start=1):
        chunk_id = f"hipertensao_chunk_{i}"
        chunks.append({
            "id": chunk_id,
            "text": sub.strip(),
            "metadata": {
                "fonte": "Identificação e Clusterização de Pacientes Hipertensos por Grau de Risco",
                "tipo_documento": "protocolo_clinico",
                "projeto": "Hipertensao",
                "chunk_index": i,
                "total_chunks": len(sub_chunks),
                "namespace": NAMESPACE,
                "arquivo_origem": filepath
            }
        })
    return chunks

def gerar_embedding(texto: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texto
    )
    return response.data[0].embedding

def upsert_chunks(chunks: list[dict], batch_size: int = 100) -> None:
    vectors = []
    for chunk in chunks:
        embedding = gerar_embedding(chunk["text"])
        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                **chunk["metadata"],
                "text": chunk["text"]
            }
        })
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch, namespace=NAMESPACE)
    print(f"Total de chunks criados: {len(chunks)}")
    print(f"Total de vetores enviados ao Pinecone: {len(vectors)}")
    print(f"Index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")

def save_jsonl(chunks: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    chunks = build_chunks(FILEPATH)
    save_jsonl(chunks, OUTPUT_JSONL)
    upsert_chunks(chunks)
