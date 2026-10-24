import json
from pathlib import Path
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

folder = Path(__file__).resolve().parent
text = (folder / "spring_caching.txt").read_text(encoding="utf-8")

# Download only Qwen's tokenizer files, NOT its model weights.
# This lets us measure chunks using the embedding model's tokens.
model_name = "Qwen/Qwen3-Embedding-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Start with 500-token passages and up to 75 tokens of overlap.
# This is a starting design choice; evaluation will test its quality.
# LangChain prefers paragraph, line, then word boundaries.
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=500,
    chunk_overlap=75,
)
chunks = splitter.split_text(text)

if not chunks:
    raise ValueError("No chunks created. Check spring_caching.txt.")

# Store each passage with its identity and citation information.
# A dictionary is similar to a Java Map; records is a list of them.
records = []
for number, chunk in enumerate(chunks, start=1):
    token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
    records.append({
        "id": f"spring-caching-{number:03d}",
        "text": chunk,
        "source": "https://docs.spring.io/spring-boot/4.0/reference/io/caching.html",
        "title": "Spring Boot 4.0 - Caching",
        "embedding_model": model_name,
        "token_count": token_count,
    })

# JSON saves the list so the embedding step can load it later.
output_file = folder / "chunks.json"
output_file.write_text(
    json.dumps(records, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

# Summing chunk sizes includes duplicated text from overlap.
# Actual API billing may differ slightly due to service tokenization.
total_tokens = sum(record["token_count"] for record in records)
estimated_cost = total_tokens / 1_000_000 * 0.01

print(f"Saved {len(records)} chunks to {output_file}")
print(f"Largest chunk: {max(r['token_count'] for r in records)} tokens")
print(f"Total chunk tokens: {total_tokens}")
print(f"Estimated embedding cost: ${estimated_cost:.8f}")
print("\nFirst chunk preview:\n", records[0]["text"][:500])