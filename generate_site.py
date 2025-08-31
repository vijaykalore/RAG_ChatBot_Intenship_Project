import os

OUT = "site_pages"
os.makedirs(OUT, exist_ok=True)

pages = {
    "index.html": "<h1>Welcome to Zep Demo Site</h1><p>This demo site contains pages about data science, Python, web scraping, and RAG.</p>",
    "python_basics.html": "<h1>Python Basics</h1><p>Python is a versatile language. Variables, lists, dicts, functions, and classes are core concepts.</p>",
    "web_scraping.html": "<h1>Web Scraping</h1><p>Web scraping uses requests and BeautifulSoup. Respect robots.txt and rate limits.</p>",
    "data_science.html": "<h1>Data Science</h1><p>Data science involves cleaning, EDA, modeling, and visualization.</p>",
    "nlp_intro.html": "<h1>Introduction to NLP</h1><p>NLP tasks include tokenization, embeddings, classification, and generation.</p>",
    "rag_overview.html": "<h1>RAG Overview</h1><p>Retrieval-Augmented Generation combines a retriever (vector DB) with a generator (LLM).</p>",
    "faiss_and_embeddings.html": "<h1>FAISS and Embeddings</h1><p>FAISS is a fast vector similarity library. Embeddings turn text into vectors.</p>",
    "deploying_apps.html": "<h1>Deploying Apps</h1><p>Common platforms: Heroku, AWS, Azure. Docker helps reproducible deployments.</p>"
}

for name, html in pages.items():
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<html><body>{html}</body></html>")

print(f"Generated {len(pages)} local HTML pages in {OUT}/")
