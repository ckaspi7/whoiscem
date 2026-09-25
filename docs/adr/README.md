# Architecture Decision Records

Each of these documents a real decision made on this project, on purpose,
with the evidence that drove it — not a rationalization written after the
fact. Several record a decision that contradicts what looked like the
obvious choice going in (classifier over a real agent; dense retrieval over
the hybrid pipeline the project is nominally built around); that's the point
of writing them down. See [CHANGELOG.md](../../CHANGELOG.md) for the numbers
behind each one and [README.md](../../README.md) for the full narrative.

| ADR | Decision |
|---|---|
| [0001](0001-classifier-stays-the-default-agent.md) | Classifier stays the default agent architecture |
| [0002](0002-dense-retrieval-ships-by-default.md) | Dense retrieval ships by default; the hybrid pipeline is measured, not removed |
| [0003](0003-local-cross-encoder-over-llm-rerank.md) | A local cross-encoder reranks, not an LLM call |
| [0004](0004-chat-model-and-agent-mode-are-orthogonal.md) | `CHAT_MODEL` and `AGENT_MODE` are independent settings; gpt-6-luna is a recommended override, not a default |
| [0005](0005-self-correction-scoped-to-resume-route.md) | The self-correction retry is scoped to the resume route only |
| [0006](0006-fail-open-judge-fail-closed-tool-errors.md) | The faithfulness judge fails open; a tool error must never reach context |
| [0007](0007-rate-limiting-and-spend-cap-live-in-a-new-api-layer.md) | Rate limiting and the spend cap live in a new API layer, not in Streamlit directly |
