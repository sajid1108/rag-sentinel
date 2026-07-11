from graph import build_graph

app = build_graph()

question = "What programming languages does this person know?"

initial_state = {
    "question": question,
    "current_query": question,
    "context": "",
    "answer": "",
    "verdict": "",
    "reason": "",
    "retry_count": 0
}

print(f"\n{'='*60}")
print(f"QUESTION: {question}")
print(f"{'='*60}\n")

final_state = app.invoke(initial_state)

print(f"\n{'='*60}")
print("FINAL ANSWER:")
print(final_state["answer"])
print(f"{'='*60}")