---
paths:
  - "chatbot.py"
  - "pages/**"
  - ".streamlit/**"
  - "tests/**/test_app_boot.py"
---

# Streamlit UI

`main()` ends every turn that just ran the graph with `st.rerun()` (so the
sidebar's cost/latency numbers update this same turn instead of lagging one
behind), which means anything rendered inside that turn's own `st.chat_message`
block is superseded almost immediately by the rerun's redraw of chat history —
a fact confirmed by testing, not assumed: an early version of this fix rendered
the reveal into that now-discarded block and it never reached a visitor at
all. The actual reveal happens in the "Display history" loop instead:
`st.session_state.last_ai_message_index` names the one message (set right
before that turn's `st.rerun()`) that gets `_fade_in_html`'s CSS-staggered
per-word fade; every other message, including that same one once a later
turn moves the index on, renders as plain markdown — otherwise old answers
would replay their fade-in on every future redraw. The per-word stagger is
CSS-timed rather than a Python-side reveal loop because Streamlit re-issues
a placeholder's entire markdown content on every update, which would replay
every earlier word's animation alongside each new one instead of only the
newest word fading in.

## Constraints

- The Streamlit app uses `st.query_params["sid"]` for session identity, making sessions shareable via URL. Phase 4.5: the value is HMAC-signed (`chatbot._sign_session_id`/`_verify_session_id`) — a `sid` that fails verification (fabricated, edited, or signed under a different `SESSION_SECRET`) is never trusted as a previously-issued session; a fresh one is minted instead. This closes fabrication, not leakage of a real signed link — see LIMITATIONS.md for that distinction.
- Streamlit Cloud secrets override `.env` values — `_apply_streamlit_secrets()` in `chatbot.py` merges them into `os.environ`, called after `set_page_config()` (reading `st.secrets` is itself a Streamlit command and must not be first) and before `setup_tracing()`/`create_assistant()`.
