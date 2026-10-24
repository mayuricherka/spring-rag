from urllib.request import urlopen
from pathlib import Path


url = "https://docs.spring.io/spring-boot/4.0/reference/io/caching.html"

with urlopen(url, timeout=30) as response:
    html = response.read().decode("utf-8")

print("Downloaded characters:", len(html))
print(html[:200])

output_path = Path(__file__).parent / "spring_caching.html"
output_path.write_text(html, encoding="utf-8")
print("Saved to:", output_path.resolve())