import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from graph import build_graph

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG Sentinel", page_icon="🛡️")
st.title("🛡️ RAG Sentinel")
st.caption("A self-healing RAG system — verifies its own answers, retries, or admits when it doesn't know.")

# --- Session state setup ---
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# --- Sidebar: PDF upload ---
with st.sidebar:
    st.header("📄 Upload a document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and st.button("Process document"):
        with st.spinner("Reading and indexing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load, split, embed
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

            # Fresh vectorstore per document (in-memory, no persistence conflicts)
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.vectorstore_ready = True
            os.remove(tmp_path)

        st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'")

# --- Main chat area ---
if not st.session_state.vectorstore_ready:
    st.info("👈 Upload a PDF to get started.")
else:
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Thinking..."):
            from graph import RAGState
            import graph as graph_module
            graph_module.vectorstore = st.session_state.vectorstore

            app = build_graph()
            initial_state = {
                "question": question,
                "current_query": question,
                "context": "",
                "answer": "",
                "verdict": "",
                "reason": "",
                "retry_count": 0
            }

            try:
                final_state = app.invoke(initial_state)
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    st.error("⏳ Gemini's free-tier daily limit has been reached (20 requests/day). Please try again later, or use a different API key.")
                    st.stop()
                else:
                    st.error(f"Something went wrong: {e}")
                    st.stop()

        st.subheader("Answer")
        st.write(final_state["answer"])

        with st.expander("🔍 See reasoning trace"):
            st.write(f"**Final verdict:** {final_state['verdict']}")
            st.write(f"**Critic's reasoning:** {final_state['reason']}")
            st.write(f"**Retries used:** {final_state['retry_count']}")