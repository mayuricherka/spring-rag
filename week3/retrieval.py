"""Read-only retrieval using the existing Week 2 model and Pinecone corpus."""

import math


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
INDEX_NAME = "spring-rag"
NAMESPACE = "spring-caching-v1"
DIMENSIONS = 4096
METRIC = "cosine"
EXPECTED_CHUNKS = 14
DOCUMENTATION_VERSION = "4.0"
SOURCE_URL = "https://docs.spring.io/spring-boot/4.0/reference/io/caching.html"
TOP_K = 5


class RetrievalError(ValueError):
    """A safe error message that contains no raw provider response."""


def validate_request(question, documentation_version):
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise RetrievalError("Enter a question of 1–2000 characters.")
    if documentation_version != DOCUMENTATION_VERSION:
        raise RetrievalError("Only Spring Boot 4.0 documentation is indexed. No embedding was requested.")


class DocumentationSearch:
    """Like a Java service with constructor-injected external clients.

    Production supplies SDK clients; tests supply mocks. Both run this same code.
    This service has no index creation, upsert, document embedding, or file writes.
    """

    def __init__(self, embedding_client, pinecone_client, *, service_mode="live_services"):
        self.embedding_client = embedding_client
        self.pinecone_client = pinecone_client
        self.service_mode = service_mode

    def search(self, question, documentation_version=DOCUMENTATION_VERSION):
        validate_request(question, documentation_version)
        try:
            # Verify the remote index BEFORE sending a paid question embedding.
            details = self.pinecone_client.describe_index(INDEX_NAME)
            if details.dimension != DIMENSIONS or details.metric != METRIC:
                raise RetrievalError("The existing index must have 4096 dimensions and cosine similarity.")
            if not details.status["ready"]:
                raise RetrievalError("The existing Pinecone index is not ready.")

            index = self.pinecone_client.Index(host=details.host)
            try:
                stats = index.describe_index_stats()
                summary = stats.namespaces.get(NAMESPACE)
                if summary is None or summary.vector_count != EXPECTED_CHUNKS:
                    raise RetrievalError("Expected 14 vectors in spring-caching-v1. Review the namespace before embedding.")

                response = self.embedding_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=[
                        "Instruct: Retrieve Spring documentation passages that answer the question.\n"
                        f"Query: {question.strip()}"
                    ],
                    encoding_format="float",
                )
                if len(response.data) != 1 or response.data[0].index != 0:
                    raise RetrievalError("Expected one question embedding.")
                vector = response.data[0].embedding
                if len(vector) != DIMENSIONS or not all(math.isfinite(value) for value in vector):
                    raise RetrievalError("The question embedding must contain 4096 finite values.")

                matches = index.query(
                    namespace=NAMESPACE,
                    vector=vector,
                    top_k=TOP_K,
                    include_metadata=True,
                    include_values=False,
                    filter={
                        "source": {"$eq": SOURCE_URL},
                        "embedding_model": {"$eq": EMBEDDING_MODEL},
                    },
                ).matches
                passages = []
                for match in matches:
                    metadata = match.metadata or {}
                    # Existing records have no version field. Verify their URL
                    # rather than invent version metadata for unrelated results.
                    if (metadata.get("source") != SOURCE_URL
                            or metadata.get("embedding_model") != EMBEDDING_MODEL
                            or not metadata.get("text")):
                        continue
                    score = float(match.score)
                    if not math.isfinite(score):
                        continue
                    passages.append({
                        "id": match.id,
                        "score": score,
                        "text": metadata["text"],
                        "source": metadata["source"],
                        "title": metadata.get("title", "Spring Boot 4.0 - Caching"),
                        "documentation_version": DOCUMENTATION_VERSION,
                        "version_basis": "stored_source_url",
                    })
                usage = response.usage
                return {
                    "status": "ok" if passages else "no_matches",
                    "service_mode": self.service_mode,
                    "embedding_model": EMBEDDING_MODEL,
                    "index": INDEX_NAME,
                    "namespace": NAMESPACE,
                    "index_settings": {
                        "dimensions": details.dimension,
                        "metric": details.metric,
                        "namespace_vectors": summary.vector_count,
                    },
                    "documentation_version": DOCUMENTATION_VERSION,
                    "version_basis": "stored_source_url",
                    "passages": passages,
                    "embedding_usage": None if usage is None else {
                        "prompt_tokens": usage.prompt_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                }
            finally:
                index.close()
        except RetrievalError:
            raise
        except Exception:
            # Provider errors may contain request data. Do not forward them.
            raise RetrievalError("Retrieval failed. Check credentials and service availability; no automatic embedding retry was made.") from None
