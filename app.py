import streamlit as st
from ingest import process_pdfs
from rag_engine import rag_chat
import os

st.title("📘 RAG Chatbot - Ask Questions From PDF")

st.write("Upload PDFs into the **pdfs/** folder, then click Process.")

# -------------------------
# PROCESS PDFs BUTTON
# -------------------------
if st.button("Process PDFs"):
    if not os.path.exists("pdfs"):
        st.error("❌ 'pdfs' folder not found! Create a folder named pdfs and add PDFs inside it.")
    else:
        msg = process_pdfs()
        st.success(msg)

# -------------------------
# ASK QUESTION
# -------------------------
query = st.text_input("Ask any question from the PDF")

if query:
    answer = rag_chat(query)
    st.write("### 📌 Answer:")
    st.write(answer)


