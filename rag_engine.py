import faiss
import numpy as np
import subprocess
import os

INDEX_PATH = "faiss_index/index.bin"
CHUNKS_PATH = "faiss_index/chunks.npy"

def run_gemma(prompt):
    """Send prompt to Gemma 2B via Ollama"""
    process = subprocess.Popen(
        ["ollama", "run", "gemma:2b"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output, _ = process.communicate(prompt)
    return output.strip()


def rag_chat(query):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return "❌ Index not found. Click PROCESS PDFs first."

    # Load FAISS & chunks
    index = faiss.read_index(INDEX_PATH)
    chunks = np.load(CHUNKS_PATH, allow_pickle=True)

    # Encode using MiniLM
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_vec = model.encode([query]).astype("float32")

    # Search
    D, I = index.search(q_vec, 3)
    context = "\n\n".join([chunks[i] for i in I[0]])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer clearly:
"""
    return run_gemma(prompt)






