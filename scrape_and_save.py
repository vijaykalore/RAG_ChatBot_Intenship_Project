import os
from bs4 import BeautifulSoup

SRC = "site_pages"
OUT = "texts"
os.makedirs(OUT, exist_ok=True)

for fname in os.listdir(SRC):
    if not fname.endswith(".html"):
        continue
    path = os.path.join(SRC, fname)
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        text = soup.get_text(separator="\n").strip()
    txt_name = os.path.splitext(fname)[0] + ".txt"
    with open(os.path.join(OUT, txt_name), "w", encoding="utf-8") as w:
        w.write(text)

print(f"Saved text files to {OUT}/")
