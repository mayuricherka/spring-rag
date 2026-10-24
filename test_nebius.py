
from getpass import getpass
from openai import OpenAI
api_key = getpass("Paste your Nebius API key, then press Return: ")
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=api_key,
    max_retries=0,
    timeout=30.0,
)

response = client.chat.completions.create(
    model="nvidia/Nemotron-3_5-Lightning",
    messages=[
        {"role": "user", "content": "In one short sentence, what is Spring caching?"}
    ],
    max_tokens=100,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},

)

print(response.choices[0].message.content)
print("Token usage:", response.usage)

