import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

TXT_DIR = "texts"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

texts = []
metadatas = []
for fname in sorted(os.listdir(TXT_DIR)):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(TXT_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    passages = [p.strip() for p in content.split("\n") if p.strip()]
    for i, p in enumerate(passages):
        texts.append(p)
        metadatas.append({"source": fname, "passage_id": i})

if not texts:
    raise SystemExit("No texts to index.")

embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
# normalize to unit vectors for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_norm = embeddings / (norms + 1e-12)

os.makedirs("vector_store", exist_ok=True)
np.save(os.path.join("vector_store", "embeddings.npy"), embeddings_norm.astype("float32"))
with open(os.path.join("vector_store", "metadata.pkl"), "wb") as f:
    pickle.dump({"texts": texts, "metadatas": metadatas}, f)

print("Built embeddings and saved to vector_store/")
