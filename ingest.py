import os
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = "pdfs"
INDEX_FOLDER = "faiss_index"
INDEX_PATH = os.path.join(INDEX_FOLDER, "index.bin")
CHUNKS_PATH = os.path.join(INDEX_FOLDER, "chunks.npy")

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def process_pdfs():
    if not os.path.exists(INDEX_FOLDER):
        os.makedirs(INDEX_FOLDER)

    all_text = ""

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            all_text += extract_text_from_pdf(pdf_path)

    if not all_text.strip():
        return "No text found in PDFs!"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(all_text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))

    return "PDFs processed successfully! FAISS index created."





