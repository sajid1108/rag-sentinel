from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 1. Load PDF
pdf_path = "data/resume.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"✅ Split into {len(chunks)} chunk(s).")

# 3. Create embeddings model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# 4. Store chunks in Chroma (creates local folder "chroma_db")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print("✅ Chunks embedded and stored in Chroma.")

# 5. Test a similarity search
query = "What programming languages does this person know?"
results = vectorstore.similarity_search(query, k=2)

print(f"\n--- Top {len(results)} result(s) for query: '{query}' ---")
for i, r in enumerate(results):
    print(f"\nResult {i+1}:")
    print(r.page_content)