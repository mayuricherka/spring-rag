import hashlib
import json
from getpass import getpass
from pathlib import Path

from openai import OpenAI
from pinecone import Pinecone

folder = Path(__file__).resolve().parent
raw = (folder / "chunks.json").read_text(encoding="utf-8")
records = json.loads(raw)
model_name = "Qwen/Qwen3-Embedding-8B"
index_name = "spring-rag"
namespace = "spring-caching-v1"

# Small-run guardrails before any paid embedding request.
if not 1 <= len(records) <= 32:
    raise ValueError("Expected 1–32 chunks for this first upload.")
if any(r["embedding_model"] != model_name for r in records):
    raise ValueError("Chunks were prepared for another embedding model.")
if sum(r["token_count"] for r in records) > 20_000:
    raise ValueError("This run exceeds our small-test token budget.")

# Verify Pinecone access and index settings BEFORE paying for embeddings.
pc = Pinecone(api_key=getpass("Pinecone API key: "))
details = pc.describe_index(index_name)
if details.dimension != 4096 or details.metric != "cosine":
    raise ValueError("Index must have 4096 dimensions and cosine metric.")
if not details.status["ready"]:
    raise ValueError("Index is not ready yet. Wait before retrying.")
index = pc.Index(host=details.host)

# A hash is a fingerprint of the input file.
# If chunks change, we must not silently reuse old embeddings.
fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()
cache_file = folder / "embeddings_cache.json"

if cache_file.exists():
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    if cache["fingerprint"] != fingerprint or cache["model"] != model_name:
        raise ValueError("Cached embeddings differ from these chunks. Stop and review.")
    vectors = cache["vectors"]
    print("Reusing cached vectors: no Nebius request.")
else:
    client = OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1/",
        api_key=getpass("Nebius API key: "),
        max_retries=0,
        timeout=60.0,
    )

    # One ordinary API request embeds all passages in this small corpus.
    response = client.embeddings.create(
        model=model_name,
        input=[r["text"] for r in records],
        encoding_format="float",
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in ordered] != list(range(len(records))):
        raise ValueError("Embedding response does not match input order.")
    vectors = [item.embedding for item in ordered]

    # Save BEFORE uploading, so an upload failure doesn't lose the vectors.
    cache_file.write_text(json.dumps({
        "fingerprint": fingerprint,
        "model": model_name,
        "vectors": vectors,
    }), encoding="utf-8")
    print("Nebius token usage:", response.usage)

if len(vectors) != len(records) or any(len(v) != 4096 for v in vectors):
    raise ValueError("Unexpected vector count or dimensions.")

# zip pairs each chunk with its corresponding vector.
# Metadata keeps the passage and source available when we retrieve it.
payload = []
for record, vector in zip(records, vectors):
    payload.append({
        "id": record["id"],
        "values": vector,
        "metadata": {
            "text": record["text"],
            "source": record["source"],
            "title": record["title"],
            "embedding_model": model_name,
        },
    })

# Upsert = insert a new ID or replace that same ID.
# A namespace keeps this corpus separate inside the index.
result = index.upsert(vectors=payload, namespace=namespace)
print(f"Pinecone accepted {result.upserted_count} vectors.")
print("Namespace:", namespace)
print("Search visibility may take a few seconds to update.")