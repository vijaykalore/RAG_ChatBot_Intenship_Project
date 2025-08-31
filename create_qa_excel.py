import pandas as pd
import os

qa = [
    ("What is RAG?", "RAG stands for Retrieval-Augmented Generation: combining a retriever (vector DB) with a generator (LLM)."),
    ("What library is good for embeddings?", "sentence-transformers is commonly used for generating text embeddings."),
    ("How to do web scraping in Python?", "Use requests to fetch pages and BeautifulSoup to parse HTML; respect robots.txt."),
    ("What is FAISS used for?", "FAISS is a library for fast similarity search over vector embeddings."),
    ("Name a common deployment platform.", "Heroku, AWS, and Azure are common deployment platforms for web apps."),
    ("What is tokenization?", "Tokenization splits text into tokens (words, subwords) used by NLP models."),
    ("How to convert text to PDF in Python?", "Use libraries like fpdf or reportlab to create PDFs programmatically."),
    ("What does EDA mean?", "EDA stands for Exploratory Data Analysis, used to understand datasets."),
    ("Which model is small & fast for embeddings?", "all-MiniLM-L6-v2 is a compact and popular sentence-transformers model."),
    ("What is the purpose of a vector index?", "A vector index enables fast nearest-neighbor search over embeddings for retrieval tasks.")
]

df = pd.DataFrame(qa, columns=["question", "answer"])
os.makedirs("data", exist_ok=True)
df.to_excel(os.path.join("data", "qa_sheet.xlsx"), index=False)
print("Saved data/qa_sheet.xlsx")
