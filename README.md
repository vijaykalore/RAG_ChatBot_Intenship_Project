# Zep Analytics — Demo RAG Chatbot Pipeline

This repository contains a small end-to-end demo pipeline that:
- Generates a local 8-page HTML site
- Scrapes the pages to plain text
- Converts the text files to PDFs
- Builds a FAISS vector index using sentence-transformers
- Creates an Excel sheet with 10 Q&A entries
- Runs a Flask chatbot that first searches the Excel Q&A (TF-IDF) and falls back to vector retrieval

Prerequisites
- Python 3.9+ (install from python.org)

Quick start (Windows PowerShell)

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_site.py
python scrape_and_save.py
python txt_to_pdf.py
python build_index.py
python create_qa_excel.py
python app.py
```

Open http://127.0.0.1:5000 in your browser and ask questions.

Notes
- This demo uses local, generated content to avoid external scraping and copyright issues.
- The Flask app does not call external LLMs. It returns retrieved passages. To add LLM summarization, include your API call in `app.py` after retrieval.
