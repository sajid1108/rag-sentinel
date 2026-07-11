from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import json

def extract_text(content):
    """Handles both plain string and list-of-dict response formats from Gemini."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

# 1. Load existing vectorstore (don't rebuild every time)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# 2. Retrieve + Generate (same as Step 6)
question = "What programming languages does this person know?"
retrieved_docs = vectorstore.similarity_search(question, k=3)
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

gen_prompt = f"""You are a helpful assistant answering questions based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

draft_answer = extract_text(llm.invoke(gen_prompt).content)
print("--- Draft Answer ---")
print(draft_answer)

# 3. CRITIC: check if the answer is grounded in the context
critic_prompt = f"""You are a strict fact-checker. Your job is to verify if the ANSWER below is fully supported by the CONTEXT.

Context:
{context}

Question: {question}

Answer to check:
{draft_answer}

Respond ONLY in this exact JSON format, nothing else:
{{
  "verdict": "PASS" or "RETRY" or "INSUFFICIENT",
  "reason": "short explanation"
}}

Rules:
- PASS: every claim in the answer is directly supported by the context.
- RETRY: the answer has unsupported claims, but better retrieval might fix it.
- INSUFFICIENT: the context fundamentally does not contain enough information to answer this question at all.
"""

critic_response = extract_text(llm.invoke(critic_prompt).content)

# Clean up response (remove markdown code fences if present)
cleaned = critic_response.strip().replace("```json", "").replace("```", "").strip()

print("\n--- Critic Verdict (raw) ---")
print(cleaned)

try:
    verdict_json = json.loads(cleaned)
    print("\n--- Parsed Verdict ---")
    print(f"Verdict: {verdict_json['verdict']}")
    print(f"Reason: {verdict_json['reason']}")
except json.JSONDecodeError:
    print("\n⚠️ Could not parse critic response as JSON.")