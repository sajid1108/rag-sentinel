import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Shared clients — created once, reused by every node
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


from typing import TypedDict, List

def extract_text(content):
    """Handles both plain string and list-of-dict response formats from Gemini."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content)


class RAGState(TypedDict):
    question: str          # original user question (never changes)
    current_query: str     # the query actually used for retrieval (may be reformulated)
    context: str            # retrieved chunks, joined as one string
    answer: str              # draft answer from the Generator
    verdict: str             # "PASS" / "RETRY" / "INSUFFICIENT"
    reason: str              # critic's explanation for the verdict
    retry_count: int         # how many retries have happened so far




def retrieve_node(state: RAGState) -> dict:
    """Searches the vector store using current_query, returns top chunks as context."""
    query = state["current_query"]
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"\n[RETRIEVE] Query: '{query}' → {len(docs)} chunk(s) found")
    return {"context": context}


def generate_node(state: RAGState) -> dict:
    """Generates a draft answer using the question + retrieved context."""
    question = state["question"]
    context = state["context"]

    prompt = f"""You are a helpful assistant answering questions based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    answer = extract_text(response.content)
    print(f"[GENERATE] Draft answer: {answer[:100]}...")
    return {"answer": answer}

import json

def critic_node(state: RAGState) -> dict:
    """Checks whether the draft answer is actually grounded in the retrieved context."""
    question = state["question"]
    context = state["context"]
    answer = state["answer"]

    critic_prompt = f"""You are a strict fact-checker. Your job is to verify if the ANSWER below is fully supported by the CONTEXT.

Context:
{context}

Question: {question}

Answer to check:
{answer}

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

    response = llm.invoke(critic_prompt)
    raw = extract_text(response.content)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(cleaned)
        verdict = result.get("verdict", "RETRY")
        reason = result.get("reason", "No reason provided.")
    except json.JSONDecodeError:
        # If the critic's output can't be parsed, fail safe: treat as RETRY
        verdict = "RETRY"
        reason = "Critic response could not be parsed."

    print(f"[CRITIC] Verdict: {verdict} — {reason}")
    return {"verdict": verdict, "reason": reason}


def rewriter_node(state: RAGState) -> dict:
    """Reformulates the search query based on why the critic rejected the answer."""
    question = state["question"]
    reason = state["reason"]
    retry_count = state.get("retry_count", 0)

    rewrite_prompt = f"""The following question was asked, but the retrieved information wasn't good enough to answer it properly.

Original question: {question}
Why it failed: {reason}

Rewrite the question as a better search query to find more relevant information.
Respond with ONLY the rewritten query, nothing else."""

    response = llm.invoke(rewrite_prompt)
    new_query = extract_text(response.content).strip()

    print(f"[REWRITER] New query: '{new_query}' (retry #{retry_count + 1})")
    return {"current_query": new_query, "retry_count": retry_count + 1}

def fallback_node(state: RAGState) -> dict:
    """Used when retries are exhausted or context is fundamentally insufficient."""
    print("[FALLBACK] Returning safe 'I don't know' response.")
    return {"answer": "I don't have enough information to answer that reliably."}

MAX_RETRIES = 2

def route_after_critic(state: RAGState) -> str:
    """Decides the next step based on the critic's verdict."""
    verdict = state["verdict"]
    retry_count = state.get("retry_count", 0)

    if verdict == "PASS":
        return "end"
    if verdict == "INSUFFICIENT":
        return "fallback"
    if verdict == "RETRY" and retry_count >= MAX_RETRIES:
        return "fallback"
    return "retry"

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("rewriter", rewriter_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end": END,
            "retry": "rewriter",
            "fallback": "fallback"
        }
    )

    graph.add_edge("rewriter", "retrieve")  # loop back
    graph.add_edge("fallback", END)

    return graph.compile()