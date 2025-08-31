import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
import pandas as pd
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)

# Load resources
MODEL_NAME = "all-MiniLM-L6-v2"
model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer as _STS
        model = _STS(MODEL_NAME)
    return model

# Load FAISS + metadata
embeddings = np.load(os.path.join("vector_store", "embeddings.npy"))
with open(os.path.join("vector_store", "metadata.pkl"), "rb") as f:
    meta = pickle.load(f)
texts = meta["texts"]

# Load QA Excel and prepare TF-IDF (if available)
qa_df = pd.read_excel(os.path.join("data", "qa_sheet.xlsx"))
if SKLEARN_AVAILABLE:
    tfidf = TfidfVectorizer().fit(qa_df["question"].tolist())
else:
    tfidf = None

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
    if tfidf is None:
        # fallback: simple substring match
        for i, q in qa_df["question"].iteritems():
            if query.lower() in str(q).lower() or str(q).lower() in query.lower():
                return {"index": int(i), "score": 1.0, "answer": qa_df.iloc[int(i)]["answer"], "question": qa_df.iloc[int(i)]["question"]}
        return {"index": -1, "score": 0.0, "answer": "", "question": ""}
    q_vec = tfidf.transform([query])
    q_mat = tfidf.transform(qa_df["question"].tolist())
    sims = cosine_similarity(q_vec, q_mat)[0]
    idx = int(np.argmax(sims))
    score = float(sims[idx])
    return {"index": idx, "score": score, "answer": qa_df.iloc[idx]["answer"], "question": qa_df.iloc[idx]["question"]}

def retrieve_vectors(query, top_k=3):
    _model = get_model()
    emb = _model.encode([query], convert_to_numpy=True)
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    # cosine similarity via dot product (embeddings already normalized)
    sims = (emb_norm @ embeddings.T)[0]
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        results.append({"score": float(sims[i]), "text": texts[int(i)]})
    return results

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_UI, resp="")

@app.route("/chat", methods=["POST"])
def chat():
    q = request.form.get("q") or request.json.get("q")
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
    vecs = retrieve_vectors(q, top_k=3)
    if not vecs:
        return jsonify({"source": "none", "answer": "No relevant content found."})

    # Combine retrieved passages into an answer (no external LLM)
    combined = "\n\n".join([f"(score:{v['score']:.3f}) {v['text']}" for v in vecs])
    return jsonify({"source": "vector", "answer": combined, "retrievals": vecs})

if __name__ == "__main__":
    app.run(debug=True)
