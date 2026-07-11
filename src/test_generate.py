from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 1. Load + split PDF
pdf_path = "data/resume.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

# 2. Embed + store (reuses existing chroma_db if already built)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# 3. Retrieve relevant chunks for a question
question = "What programming languages does this person know?"
retrieved_docs = vectorstore.similarity_search(question, k=3)
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

print("--- Retrieved Context ---")
print(context)
print("\n" + "="*50 + "\n")

# 4. Build a prompt that forces grounding in the context
prompt = f"""You are a helpful assistant answering questions based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

# 5. Generate the answer
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)
response = llm.invoke(prompt)

print("--- Generated Answer ---")
print(response.content)