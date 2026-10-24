import hashlib
import json
from getpass import getpass
from pathlib import Path
from time import perf_counter

from openai import OpenAI
from pinecone import Pinecone
from app import build_app  # Reuse our workflow without starting its input loop.

folder = Path(__file__).resolve().parent
output_file = folder / "evaluation_results_v2.json"

# Protect previous results and avoid accidentally repeating paid tests.
if output_file.exists():
    raise SystemExit("Evaluation file already exists. Review it before rerunning.")

# These are tests of our CURRENT one-document corpus.
# Cross-section questions are not the same as cross-document questions.
tests = [
    ("normal", "How do I enable caching in Spring Boot?"),
    ("normal", "What does @Cacheable do when the requested value is cached?"),
    ("normal", "Which cache provider is used if I add no specific cache library?"),
    ("normal", "What does spring-boot-starter-cache provide?"),
    ("normal", "How can I explicitly select a cache provider?"),
    ("normal", "How do I set Redis cache time-to-live to 10 minutes?"),
    ("normal", "Why should I keep Redis cache key prefixes enabled?"),
    ("normal", "How can I configure Caffeine cache settings?"),
    ("cross-section", "Compare the default simple cache provider with Redis caching."),
    ("cross-section", "How do I enable caching and then configure Redis cache expiration?"),
    ("ambiguous", "How do I change the timeout?"),
    ("ambiguous", "Which cache is best for my application?"),
    ("unsupported", "What is my company's production Redis password?"),
    ("unsupported", "What cache hit rate did my application have yesterday?"),
    ("unsupported", "What is my team's approved monthly Redis hosting budget?"),
]

pc = Pinecone(api_key=getpass("Pinecone API key: "))
index = pc.Index(host=pc.describe_index("spring-rag").host)
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=getpass("Nebius API key: "),
    max_retries=0,
    timeout=60.0,
)
app = build_app(client, index)

# Record which version of app.py produced these results.
report = {
    "app_sha256": hashlib.sha256((folder / "app.py").read_bytes()).hexdigest(),
    "corpus": "Spring Boot 4.0 caching page; 14 chunks",
    "planned_tests": len(tests),
    "results": [],
}

for number, (category, question) in enumerate(tests, start=1):
    print(f"Running {number}/{len(tests)}: {question}")
    start = perf_counter()
    try:
        result = app.invoke({"question": question})
        row = {
            "id": number,
            "category": category,
            "question": question,
            "answer": result["answer"],
            "passages": result["passages"],
            "seconds": round(perf_counter() - start, 2),
            "status": "completed",
            # Human review comes next; don't invent quality scores.
            "retrieval_score": None,
            "faithfulness_score": None,
            "notes": "",
        }
    except Exception as error:
        row = {
            "id": number,
            "question": question,
            "status": "error",
            "error_type": type(error).__name__,
        }

    report["results"].append(row)
    # Save after EVERY question, so completed results survive interruption.
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if row["status"] == "error":
        print("Stopped on error; completed results were saved.")
        break

completed = sum(r["status"] == "completed" for r in report["results"])
print(f"Completed {completed}/{len(tests)} tests.")
print("Saved:", output_file)