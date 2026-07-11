from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Path to the PDF
pdf_path = "data/resume.pdf"

# 1. Load the PDF
loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"✅ Loaded {len(pages)} page(s) from PDF.")
print("\n--- First 300 characters of page 1 ---")
print(pages[0].page_content[:300])

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)

print(f"\n✅ Split into {len(chunks)} chunk(s).")
print("\n--- First chunk ---")
print(chunks[0].page_content)