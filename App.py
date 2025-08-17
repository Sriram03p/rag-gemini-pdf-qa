import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
from PyPDF2 import PdfReader

# ------------------------------
# CONFIGURE GEMINI API
# ------------------------------
GEMINI_API_KEY = "your api key"  # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Models
EMBED_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-1.5-flash"

# ------------------------------
# FUNCTIONS
# ------------------------------
def extract_text_from_pdf(pdf_file):
    """Extract raw text from uploaded PDF."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(texts):
    """Generate embeddings using Gemini."""
    embeddings = []
    for txt in texts:
        result = genai.embed_content(model=EMBED_MODEL, content=txt)
        embeddings.append(result["embedding"])
    return np.array(embeddings, dtype="float32")

def search_index(query, index, chunks, k=3):
    """Search FAISS index for most relevant chunks."""
    q_emb = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
    q_emb = np.array([q_emb], dtype="float32")
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

def ask_gemini(query, context_chunks):
    """Ask Gemini using retrieved context."""
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are a helpful assistant. Use the following PDF context to answer:

    CONTEXT:
    {context}

    QUESTION:
    {query}

    If answer is not in the context, say 'I could not find the answer in the PDF.'
    """
    response = genai.GenerativeModel(LLM_MODEL).generate_content(prompt)
    return response.text

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("📄 PDF Q&A")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.info("Extracting and indexing PDF... please wait")

    # Step 1: Extract
    raw_text = extract_text_from_pdf(uploaded_file)

    # Step 2: Chunk
    chunks = chunk_text(raw_text, chunk_size=300)

    # Step 3: Embed & Build Index
    embeddings = embed_text(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    st.success("PDF indexed! You can now ask questions.")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        # Step 4: Retrieve
        retrieved_chunks = search_index(query, index, chunks, k=3)

        # Step 5: Ask Gemini
        answer = ask_gemini(query, retrieved_chunks)

        st.subheader("💡 Answer")
        st.write(answer)

        with st.expander("📑 Retrieved Context"):
            for c in retrieved_chunks:
                st.write(c)
