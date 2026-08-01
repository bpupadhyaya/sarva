# Chapter 5 — Memory: Sessions and Semantic Recall

Chapter 4 covered what a conversation is made of. This chapter is about
what happens to it after the run ends — `core/sarva/memory/`, which has
two layers, deliberately kept separate.

## Session persistence: plain files

`sarva.memory.session.SessionStore` is a saved conversation — one JSON
file per session name, human-readable and greppable (`cat
~/.sarva/sessions/default.json` just works). It answers "what did we
talk about," reconstructed exactly — for both tool-free conversations
(`sarva chat --session ...`) and tool-using ones (`sarva run --session
...`). The latter isn't just "the final answer": `AgentLoop.run(
transcript_out=...)` extends a caller-supplied list in place with the
*complete* message history for the run, including every intermediate
tool-call/tool-result round, not only the last assistant turn —
`RunDoneEvent.final_message` alone could never carry that, since it's
only ever the last turn. Both CLI commands build a `transcript_out`
list and hand it straight to `SessionStore.save()`, so resuming a saved
tool-using session actually restores the full back-and-forth, not a
summary of it.

**Written with owner-only permissions (`0700` directory, `0600` files),
not the platform default.** The same real gap found in
`sarva.config`'s credential file, checked for here too since a saved
session can hold real tool-use output — file contents `ReadFileTool`
read, `RunShellTool` command output, anything the user typed — at
least as sensitive as an API key: confirmed with a real `stat()` call
that `SessionStore` was leaving files at `0644`/the directory at
`0755` on this machine's real umask. `SessionStore.__init__` now
`chmod`s the sessions directory to `0700` (self-healing one an older
version already created looser), and `save()` creates each file via
`os.open(..., 0o600)` directly rather than `Path.write_bytes`'s
platform-default mode, with an explicit `chmod` afterward too so an
existing insecurely-written file gets tightened on its next save.
POSIX-only in practice, the same honesty this project already applies
to the Windows TTS and credential-file gaps — `os.chmod` doesn't give
real per-user isolation on Windows.

**An invalid `--session`/`session` name used to crash instead of
failing cleanly.** `_sanitize()`'s own `ValueError` (a genuinely good,
actionable message — "use only letters, digits, '-', and '_'") was
never actually caught anywhere it could be reached from a real user
action: `sarva chat --session "bad name!"`/`sarva run`/`sarva sessions
clear` all crashed with a raw Python traceback, and both `POST /chat`
and `/ws/chat` had the identical gap — the REST case a genuine
unhandled `500`, and the WebSocket case worse still: no error frame at
all, just a bare `ClosedResourceError` on the client's next read,
confirmed directly with a real `TestClient` WebSocket session before
this fix. The same "raw exception instead of a clean, actionable
error" bug class already fixed for `eval`/`distill`'s unknown `--model`
handling, just never checked for the one other place a caller-supplied
string reaches an internal validator. Fixed at every real call site: a
shared `_load_session_history` helper for `chat`/`run`, a direct
`try`/`except` in `sessions clear`, and — for the two server
endpoints — reported as a real `state=failed` result with the actual
reason in `detail`, the identical shape an unknown `--model` already
produces, so `/ws/chat` clients (including the desktop app, via the
`state_changed`-detail fix from a few milestones back) show it with no
client-side changes needed.

**`save()` itself used to be able to destroy a previously-good session
on an interrupted write — a real bug found by actually simulating one,
not a theoretical concern.** The permissions fix above wrote the new
content via `os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
0o600)` — directly against the real session file. `O_TRUNC` truncates
the file to 0 bytes the instant it's opened, *before* a single byte of
the new content is written. A crash between that `open()` and the
write completing (an OOM-kill, a `SIGKILL`, a real power loss) left the
file empty — confirmed live: a valid, real 150-byte session file became
0 bytes, and the next `load()` raised a pydantic `ValidationError` on
history that used to be perfectly fine. Every prior "corrupted-on-disk-
state" fix in this project (this file's own config/session/checkpoint/
vector-memory fixes) made *reading* an already-corrupted file fail
cleanly — none of them stopped `save()` itself from being the thing
that turns a good file into a corrupt one. Fixed the standard way:
write the new content to a sibling temp file first, then `os.replace()`
into place — atomic on both POSIX and Windows, so the real path always
holds either the last fully-written version or the brand new one,
never a partial one. `sarva.config.save_config`/`unset_config` shared
the identical bug (same `os.open(..., O_TRUNC)` pattern against the
real credential file) and got the identical fix, factored into one
shared `_write_config` helper rather than duplicated across both
functions. Verified the new tests are real: simulating a crash right
before the atomic rename step (by making `os.replace()` itself raise)
and confirming the real file still holds its previous, complete
content afterward — reverting the fix and re-running the same test
confirms it fails because the reverted code never calls `os.replace()`
at all.

**That fix was never centralized, so it never propagated past the two
places it started.** A later sweep of this codebase — specifically
looking for "other duplicated logic that may have drifted the same way"
after finding an independent, unpatched copy of a different bug
elsewhere — found three more real call sites writing valuable,
hard-to-regenerate data with the exact same unfixed pattern:
`sarva.agent.tools.WriteFileTool` (writing arbitrary real user files on
essentially every agent file-editing turn), `sarva.distill.save_jsonl`
(distillation records that cost real provider API calls to generate),
and the `sarva_foundry` checkpoint/tokenizer save paths (a training
run's actual GPU-hours of progress). `sarva.atomic_write` now holds the
one shared implementation `sarva.config`/`sarva.memory.session` and
these three sites all call, instead of each independently reinventing
it (or, as had actually happened, forgetting to). See
`sarva_foundry.atomic_write` (mirrored, not imported — `core` and
`sarva_foundry` share no dependency in either direction) for the
training-side equivalent, covering `Trainer.save_checkpoint`,
`ByteLevelBPETokenizer.save`, and the checkpoint bundle's `config.json`
write in `sarva.providers.foundry_provider.save_checkpoint_bundle`.

**A sixth site, missed by that sweep, found by a follow-up one:**
`sarva.cli._write_bytes_or_exit` (backing `speak --out`'s own write)
had the identical unfixed pattern — see the packaging chapter.

**The shared helper itself had a real thread-safety bug, found by a
later sweep specifically re-checking the propagation fix's own
infrastructure rather than only its call sites.** The sibling temp
filename was built from `os.getpid()` alone — unique across processes,
but identical across every *thread* of one process, since a PID names
a process, not a thread. Two real `threading.Thread`s calling
`atomic_write`/`atomic_write_bytes`/`atomic_write_text` on the *same*
path concurrently raced the same temp file: whichever thread's
`os.replace()` ran second found the temp path already renamed away by
the winner and raised an uncaught `FileNotFoundError`, confirmed live,
deterministically, 10/10 trials. Not reachable through any current
call site today (each one either serializes writes some other way
already, or isn't invoked from two genuinely concurrent threads yet),
but this module's own docstring invites every future writer of real
data to reach for it without re-auditing the file — so the helper
itself needs to be correct under concurrency, not just its current
callers. Fixed by including `threading.get_ident()` in the temp
filename too (unique among the threads alive at any one moment, unlike
a PID), in both `sarva.atomic_write` and its `sarva_foundry` mirror.

**A different, higher-level race lived one layer up, in the server's
own turn handling — a real lost-update bug on concurrent turns against
the same session.** `POST /chat` and `/ws/chat` each do `store.load(
session)` at the start of a turn, run the full `AgentLoop.run()` (real,
possibly multi-second model-provider calls in between), then `store.
save(session, transcript)` with the *whole* new history at the end —
the same unlocked read-modify-write shape already fixed for `config.
json`'s own concurrent-write race, just never applied to session
storage. Confirmed live, both with real `threading.Thread`s and (more
importantly, since it's the server's actual concurrency model) with
plain `asyncio.gather` over two coroutines mirroring `/chat`'s own
code path — no threads needed at all: two ordinary turns on the *same*
session name (two browser tabs both chatting into the default session,
say) produced a session file holding only one turn's new message, the
other silently discarded, no error to either client. Fixed with a
per-session `asyncio.Lock` (`_locked_session` in `sarva.server.app`),
held for the *entire* load-through-save span of a turn, not just the
final write — locking only the save would still let both turns load
the same stale history before either saved, reproducing the identical
race. Turns on different sessions are fully unaffected, since each
session gets its own lock; two turns on the *same* session now
serialize, the only sane semantics for what is, after all, one linear
conversation. **Deliberately in-process only, named honestly rather
than oversold:** a real agent turn awaits genuine multi-second
provider calls, and a cross-process file lock (the `config.json`
approach) held for that whole span would need its own careful async-
compatible wait implementation to avoid blocking the event loop — a
CLI process and a running server (or two CLI invocations) writing the
same session file concurrently is the identical bug shape but a
structurally different, still-open case, not covered by this fix.
Verified with a real `httpx.AsyncClient` over the app's own ASGI
transport (not the synchronous `TestClient` used elsewhere, which
can't genuinely interleave two in-flight requests) firing two truly
concurrent `POST /chat` calls at the same session — reverting the fix
reproduced the exact loss (`got 2` messages instead of the expected 4)
before re-applying.

## Semantic memory: TF-IDF + cosine similarity

`sarva.memory.vector.VectorMemoryStore` answers a different question:
"what do I already know that's *relevant* to this new thing," across
however many past notes have accumulated — a search problem, not a
reconstruction problem. This is exactly what `sarva.memory`'s own module
docstring named as future work from the start: "a vector index or
database-backed store can layer on top later without changing this
contract." Layered on top — `session.py` is completely untouched.

**The same file-permission sweep that fixed `session.py` found this
store's SQLite file (`~/.sarva/memory.db`) at the identical `0644`/
`0755` gap** — `remember`/`recall_memory` can hold text just as
sensitive as a saved session. Fixed slightly differently here, and
actually more completely: `VectorMemoryStore.__init__` `chmod`s the
parent directory to `0700` *before* `sqlite3.connect()` ever creates
the database file, so there's no window at all where another local
user could reach the file path, then `chmod`s the file itself to
`0600` too (both for defense in depth and to tighten a DB an older
version already wrote insecurely).

**A real bug found by actually writing garbage bytes to a real
`memory.db` path, the fourth instance of the "corrupted on-disk state"
bug class already fixed for `~/.sarva/config.json`, a saved session
file, and a foundry checkpoint bundle:** `sqlite3.connect()` itself
never fails on a bad file — connections are lazy — so the real error
only surfaces on the first actual query, the `CREATE TABLE IF NOT
EXISTS` this constructor already runs. That raised a raw, uncaught
`sqlite3.DatabaseError` — confirmed with two distinct real corruption
modes: a file that's genuinely "not a database" at all, and a real,
previously-valid database truncated mid-file (both produce the same
exception class, just different messages, so one `except` clause
covers both). Lower severity than the other three instances of this
bug class: both real callers (`RememberTool`/`RecallMemoryTool`) only
ever reach construction through `AgentLoop.run()`'s tool-dispatch
`except Exception`, so it never crashed a live agent turn — but a
direct caller of `VectorMemoryStore` outside that wrapper still got a
leaky, undocumented `sqlite3.DatabaseError` instead of one clean
exception type. Fixed with a new `MemoryStoreError`, raised from
`__init__` in place of the raw sqlite3 exception.

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

## Build it yourself

- `sarva chat` runs with an empty tool list (`tools=[]`) — memory tools
  are only available via `sarva run`, which wires in `BUILTIN_TOOLS`.
  With a real model configured (`ANTHROPIC_API_KEY` set — the offline
  Mock provider just echoes text back and never decides to call a tool
  on its own, confirmed by actually running it: `sarva run "remember
  that my favorite color is teal" --session demo --auto` against Mock
  produces a plain echo, not a `remember` call), run `sarva run
  "remember that my favorite color is teal" --session demo`, then in a
  fresh call `sarva run "what's my favorite color?" --session demo` —
  no code needed, just the CLI, to see both layers work together (the
  model calling `remember`, then a later turn calling `recall_memory`
  and getting back exactly what it stored).
- Try the same with a *different* `--session` name and confirm the
  second session genuinely can't see the first's memory — the
  per-session isolation this chapter describes, not assumed to hold.
- Read `tests/conformance/test_vector_memory.py`'s
  `test_search_ranks_the_topically_relevant_entry_first` — it doesn't
  just check that search returns *something*, it confirms a
  topically-related stored note actually outscores an unrelated one for
  a matching query, a real property of the TF-IDF + cosine similarity
  math, not a placeholder assertion.
- `cat ~/.sarva/sessions/<name>.json` after a real `sarva run --session
  ...` with tool calls in it, and see the full transcript — tool calls
  and results included — sitting there as plain, readable JSON.
