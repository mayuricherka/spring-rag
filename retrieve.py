from getpass import getpass
from openai import OpenAI
from pinecone import Pinecone

# Connect to the index we already populated. No documents are re-uploaded.
pc = Pinecone(api_key=getpass("Pinecone API key: "))
details = pc.describe_index("spring-rag")
index = pc.Index(host=details.host)

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=getpass("Nebius API key: "),
    max_retries=0,
    timeout=30.0,
)

question = input("Your Spring caching question: ").strip()
if not question or len(question) > 2000:
    raise ValueError("Enter a question between 1 and 2000 characters.")

# Qwen recommends a retrieval instruction for QUESTIONS.
# Document passages were embedded as plain text.
query_text = (
    "Instruct: Retrieve Spring documentation passages that answer the question.\n"
    f"Query: {question}"
)

# Embed only the question, using the SAME model as the documents.
response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-8B",
    input=[query_text],
    encoding_format="float",
)
question_vector = response.data[0].embedding

# top_k=3 asks for the three most similar passages.
# Metadata contains the original text and citation URL.
results = index.query(
    namespace="spring-caching-v1",
    vector=question_vector,
    top_k=3,
    include_metadata=True,
    include_values=False,  # Don't return thousands of vector numbers.
)

if not results.matches:
    print("No passages found. Check index/namespace and upload visibility.")

for rank, match in enumerate(results.matches, start=1):
    metadata = match.metadata or {}
    print(f"\n--- Result {rank}: {match.id} ---")
    print(f"Cosine similarity: {match.score:.4f}")
    print("Source:", metadata.get("source", "Unknown"))
    print(metadata.get("text", "No text stored"))

print("\nQuestion embedding usage:", response.usage)