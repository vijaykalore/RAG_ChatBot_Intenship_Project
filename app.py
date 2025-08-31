import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load corpus passages (built by build_index.py -> metadata.pkl)
with open(os.path.join("vector_store", "metadata.pkl"), "rb") as f:
    meta = pickle.load(f)
texts = meta.get("texts", [])

# TF-IDF over corpus texts for retrieval
corpus_vectorizer = TfidfVectorizer().fit(texts) if texts else None
corpus_tfidf = corpus_vectorizer.transform(texts) if texts else None

# Load QA Excel and prepare TF-IDF
qa_df = pd.read_excel(os.path.join("data", "qa_sheet.xlsx"))
qa_vectorizer = TfidfVectorizer().fit(qa_df["question"].tolist())
qa_q_mat = qa_vectorizer.transform(qa_df["question"].tolist())

HTML_UI = """
<!doctype html>
<title>Zep Demo Chatbot</title>
<h2>Zep Demo Chatbot</h2>
<form action="/chat" method="post">
  <input name="q" style="width:600px" placeholder="Ask a question..." />
  <input type="submit" value="Send" />
</form>
<div id="resp">{{resp}}</div>
"""

def search_excel(query, top_k=1):
    q_vec = qa_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, qa_q_mat)[0]
    idx = int(np.argmax(sims))
    score = float(sims[idx])
    return {"index": idx, "score": score, "answer": qa_df.iloc[idx]["answer"], "question": qa_df.iloc[idx]["question"]}

def retrieve_corpus(query, top_k=3):
    if corpus_vectorizer is None:
        return []
    q_vec = corpus_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, corpus_tfidf)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in idxs:
        results.append({"score": float(sims[i]), "text": texts[int(i)]})
    return results

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_UI, resp="")

@app.route("/chat", methods=["POST"])
def chat():
    q = request.form.get("q") or (request.json and request.json.get("q"))
    if not q:
        return jsonify({"error": "No question provided"}), 400

    # 1) Try Excel QA
    excel_res = search_excel(q)
    if excel_res["score"] >= 0.45:
        return jsonify({
            "source": "excel",
            "matched_question": excel_res["question"],
            "score": excel_res["score"],
            "answer": excel_res["answer"]
        })

    # 2) Fallback to vector retrieval
    vecs = retrieve_corpus(q, top_k=3)
    if not vecs:
        return jsonify({"source": "none", "answer": "No relevant content found."})

    # Combine retrieved passages into an answer (no external LLM)
    combined = "\n\n".join([f"(score:{v['score']:.3f}) {v['text']}" for v in vecs])
    return jsonify({"source": "vector", "answer": combined, "retrievals": vecs})

if __name__ == "__main__":
    app.run(debug=True)
