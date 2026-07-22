# Memory: sessions and semantic recall

`core/sarva/memory/` has two layers, deliberately kept separate.

## Session persistence: plain files

`sarva.memory.session.SessionStore` is a saved conversation — one JSON
file per session name, human-readable and greppable (`cat
~/.sarva/sessions/default.json` just works). It answers "what did we
talk about," reconstructed exactly.

## Semantic memory: TF-IDF + cosine similarity

`sarva.memory.vector.VectorMemoryStore` answers a different question:
"what do I already know that's *relevant* to this new thing," across
however many past notes have accumulated — a search problem, not a
reconstruction problem. This is exactly what `sarva.memory`'s own module
docstring named as future work from the start: "a vector index or
database-backed store can layer on top later without changing this
contract." Layered on top — `session.py` is completely untouched.

### Why TF-IDF, not neural embeddings

A real neural-embedding pipeline needs a live embedding-model API. This
project has no configured embeddings provider — and Sarva's
provider-agnostic design (§3.1) means this store shouldn't hard-code one
in — so building against a specific embeddings API right now would be
unverifiable without credentials this environment doesn't have. That's
the same trap a web-search tool would fall into, which is why this
entry built a memory store instead: something genuinely testable, fully
offline, today.

TF-IDF is the honest first tier instead, and it's not a toy stand-in —
it's a real technique with real math: each document becomes a sparse
*vector* (one weighted dimension per distinct term, not a dense neural
one), and relevance is scored with a real *cosine similarity* —
precisely the same metric dense-embedding retrieval uses, just over a
different kind of vector. `VectorMemoryStore` stores raw text in SQLite
and computes TF-IDF vectors per query (IDF weights recomputed over
exactly the candidate set being searched, so a session-scoped search
isn't polluted by unrelated sessions' vocabulary), rather than
`sqlite-vec` (the design doc's stated tech choice for *dense* vector
ANN search at scale — not the right tool for sparse, exactly-scored
vectors at this project's memory-store size). A real embedding-provider
tier can slot in alongside this later without changing the storage
contract.

### Wired into the agent, honestly scoped

`RememberTool` and `RecallMemoryTool` (`core/sarva/agent/tools.py`) put
this in `BUILTIN_TOOLS`, so the model can choose to save a note and
later search for it — both explicit tool calls, not a hidden background
process that silently logs every turn. The default store is opened
*lazily*, on first actual use, not at construction: `BUILTIN_TOOLS` is a
module-level list, so eagerly opening a database connection in
`__init__` would make merely *importing* `sarva.agent.tools` create a
real file at `~/.sarva/memory.db` on every machine that imports it —
caught and fixed before shipping, not after.

### Real per-session isolation

`ToolContext` carries an optional `session_id`, threaded from
`AgentLoop.run(session_id=...)` — which the CLI's `--session` flag and
the server's `session` request field both populate directly.
`RememberTool`/`RecallMemoryTool` prefer `ctx.session_id` over their own
constructor-time default, so two different `sarva chat --session work`
and `sarva chat --session personal` conversations get genuinely separate
memories, not a shared `"default"` bucket — verified end to end with a
tool that echoes `ctx.session_id` back through a real loop run, not just
checked that the parameter exists. A run with no session at all
(`sarva chat` with no `--session`) leaves `ctx.session_id` as `None` and
falls back to the tool's own default, exactly as before this was wired
in — every existing call site that never sets a session is unaffected.
