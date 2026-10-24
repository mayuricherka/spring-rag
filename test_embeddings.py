from getpass import getpass
from openai import OpenAI

# Hidden input: paste your existing Nebius key in the terminal.
# Never put the actual key in this file or share it in chat.
api_key = getpass("Nebius API key: ")

# Configure an HTTP client, similar to a Java REST client.
# The OpenAI SDK can talk to Nebius's compatible API.
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=api_key,
    max_retries=0,  # Avoid automatic repeat requests during testing.
    timeout=30.0,
)

# This line sends ONE paid request to Nebius.
# input is a list, so we can embed multiple passages later.
response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-8B",
    input=["Spring caching stores method results for reuse."],
    encoding_format="float",  # Return numbers directly.
)

# data[0] is the result for the first input passage.
# An embedding is a vector: a list of floating-point numbers.
vector = response.data[0].embedding

print("Vector dimensions:", len(vector))
print("First five numbers:", vector[:5])
print("Token usage:", response.usage)

# Estimate this request's embedding cost using your verified rate.
if response.usage is not None:
    cost = response.usage.prompt_tokens / 1_000_000 * 0.01
    print(f"Estimated embedding cost: ${cost:.8f}")