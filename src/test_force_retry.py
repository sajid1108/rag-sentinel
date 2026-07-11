from graph import critic_node, rewriter_node, RAGState

# Simulate a scenario where the context is real, but the generator hallucinated
fake_state: RAGState = {
    "question": "What programming languages does this person know?",
    "current_query": "What programming languages does this person know?",
    "context": "Technical Skills\n• Languages: Python, SQL, Java, C++, Bash/Linux.",
    "answer": "This person knows Python, SQL, Java, C++, Bash/Linux, and is also fluent in Rust and Go.",
    "verdict": "",
    "reason": "",
    "retry_count": 0
}

print("--- Simulated hallucinated answer ---")
print(fake_state["answer"])
print()

# Run critic on this deliberately wrong answer
critic_result = critic_node(fake_state)
fake_state.update(critic_result)

print()
print(f"Verdict: {fake_state['verdict']}")

# If critic correctly catches it, run the rewriter to prove that path too
if fake_state["verdict"] == "RETRY":
    rewriter_result = rewriter_node(fake_state)
    fake_state.update(rewriter_result)
    print()
    print("--- Rewriter fired successfully ---")
    print(f"New query: {fake_state['current_query']}")
    print(f"Retry count: {fake_state['retry_count']}")
else:
    print("\n⚠️ Critic did not flag this as RETRY — unexpected for this test.")