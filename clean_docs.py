from pathlib import Path
from bs4 import BeautifulSoup

# Path is similar to Java's java.nio.file.Path.
# __file__ is this script's location; .parent is its folder.
folder = Path(__file__).resolve().parent
input_file = folder / "spring_caching.html"
output_file = folder / "spring_caching.txt"

# Read our downloaded page instead of downloading it again.
html = input_file.read_text(encoding="utf-8")

# Parse HTML into a tree so we can select and remove elements.
soup = BeautifulSoup(html, "html.parser")
article = soup.find("article")

# None is Python's equivalent of Java's null.
# Stop rather than accidentally indexing the wrong page content.
if article is None:
    raise ValueError("No article found. Check the saved HTML.")

# Remove navigation and the latest-version banner, not the
# actual documentation tips, warnings, headings, or examples.
# select() uses CSS selectors; a leading dot means a CSS class.
for element in article.select(
    "nav, script, style, .breadcrumbs-container, "
    ".admonitionblock.latest"
):
    element.decompose()

# Code examples can contain nested HTML spans.
# Flatten each <pre> into one text node first so extraction
# doesn't insert extra line breaks inside the code.
for code_block in article.find_all("pre"):
    code_text = code_block.get_text()
    code_block.clear()
    code_block.append(code_text)

# Extract readable text, separating HTML text nodes by newlines.
# This is a simple text export, not a perfect Markdown converter.
text = article.get_text(separator="\n", strip=True)

if not text.strip():
    raise ValueError("The extracted article is empty.")

# Save the FULL article. Only the preview below is shortened.
output_file.write_text(text, encoding="utf-8")

# f-strings insert variable values, like Java string formatting.
print(f"Saved {len(text):,} characters to {output_file}")
print("\nPreview:\n")
print(text[:700])  # Python slice: first 700 characters only.