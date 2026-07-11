import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found. Check your .env file.")
    exit()

print("✅ API key loaded successfully.")

# Test a simple call to Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

response = llm.invoke("Say 'Hello, RAG Sentinel is alive!' and nothing else.")
print("Gemini response:", response.content)