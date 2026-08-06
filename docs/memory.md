# Chapter 5 — Memory: Sessions, Semantic Recall, and Long-Term Notes

Chapter 4 covered what a conversation is made of. This chapter is about
what happens to it after the run ends — `core/sarva/memory/`, which has
three layers, deliberately kept separate: session persistence (this
conversation's own history), semantic recall (session-scoped notes,
found by meaning), and long-term memory (durable notes visible across
every conversation, found by exact text). The third layer was a real,
named-but-unbuilt gap for a long time — closed further down this
chapter.

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

**`_sanitize()`'s character check said nothing about length — a much
later fresh-eyes sweep found the identical "raw OS error instead of a
clean validation error" shape the long-term-memory topic-name fix
below closes for a completely different tier.** A session name past
the filesystem's max filename length reached `os.open()` (inside
`locked()`'s `_acquire`) and raised a raw `OSError` (`ENAMETOOLONG`)
— a completely different exception type than the `ValueError` every
real call site above catches, since that was the only exception type
`_sanitize()` was ever documented to raise. Confirmed live: `POST
/chat` with a 300-character session name crashed with a raw 500
(`OSError: [Errno 63] File name too long`); `/ws/chat`'s equivalent
crash was worse — the ASGI call died with no frame sent at all, not
even the clean failure pair this handler sends for every other
validation error covered above. `ChatRequest.session` has no length
constraint and the WS frame is raw schema-less JSON, so this is
reachable through the server's real public REST/WS surface — a buggy
session-id generator, a proxy that mangles a session parameter, or
simply a long user-typed name, no adversarial intent needed. Fixed at
the source, in `_sanitize()` itself, with a 200-character limit
(comfortably under every common filesystem's ~255-byte filename limit
even after the `.lock`/`.json` suffix, and an exact byte bound since
`_VALID_NAME`'s charset is single-byte ASCII) — every existing `except
ValueError` already in place at the call sites above gets this for
free, rather than patching each one individually. Verified live after
the fix: the same request now returns a clean `state=failed` with
`"session name too long (300 chars, max 200): ..."`. Verified by
reverting and watching the new tests fail with the literal old bug's
own exception reproducing itself: `OSError: [Errno 63] File name too
long`. 2 new tests, 733 → 735 Python tests.

**`$` in `_VALID_NAME` doesn't mean "end of string" — a much later
fresh-eyes sweep found a single trailing newline silently bypassed the
whole allowlist.** Python's `re` documents `$` as matching "at the end
of the string, or just before the newline at the end of the string" —
not a genuine end-of-string anchor. `_VALID_NAME` was built with `$`,
so a session name with exactly one trailing `"\n"` appended (e.g.
`"default\n"`) matched the pattern, and `_sanitize()` returned the
string unchanged — silently bypassing the documented "use only
letters, digits, '-', and '_'" allowlist for that one shape. Not
cosmetic: `_sanitize()` feeds straight into `_path()`'s filename, so
`"default\n"` produced a real, distinct on-disk file
(`` `default\n.json` ``, not `default.json`) — silently forking the
session's history under a name a caller round-tripping through
`load("default")` could never see again. Reachable through the real
public surface with no adversarial intent: `ChatRequest.session` has
no content validator beyond this function, and a trailing `"\n"` is an
extremely common artifact of reading a session id from a file or pipe
without stripping it. Confirmed live: `_sanitize("session\n")` returned
`"session\n"` unchanged instead of raising, and saving under that name
then reading back `load("default")` returned an empty history — the
session's real content silently stuck under the newline-suffixed file.
Fixed by switching to `\Z`, the true end-of-string anchor with no
newline exception — confirmed live every other already-tested
valid/invalid name is unaffected; only the single-trailing-newline
shape changes from accepted to rejected. Verified by reverting and
watching the new test fail with the literal old bug's own shape: `DID
NOT RAISE ValueError`. 1 new test, 847 → 848 Python tests.

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

**The same helper also leaked its own temp file on every write
failure — a real bug found by a later sweep applying the exact same
lens (re-examine the mechanism itself) a third round running.** A
crash-safe *target* file (the whole point of this module) still left
its sibling *temp* file behind, permanently, whenever `write_fn`
itself raised partway through — nothing anywhere ever cleaned it up.
Confirmed live: a `write_fn` that wrote 15000 bytes then raised left
exactly that file sitting on disk after the exception propagated.
Worst for `sarva_foundry.train.trainer.Trainer.save_checkpoint`
(multi-GB checkpoint writes during real training runs) — a disk-full
failure there is a realistic scenario this bug made actively worse,
not just unhandled: the leaked partial checkpoint consumes disk space,
making the *next* save more likely to hit the same disk-full failure
again, the underlying cause compounding itself. Fixed with a
`try`/`except`/`finally`-shaped cleanup in `atomic_write` itself: on
any exception from `write_fn`, the temp file is removed (best-effort —
a failure to remove it doesn't mask or replace the original exception)
before re-raising, in both `sarva.atomic_write` and its `sarva_foundry`
mirror.

**That fix only covered `write_fn` failing, not `os.replace()` itself —
a real bug found by a third sweep of this same module, deliberately
re-reading it line by line because it had already had two real bugs
found in it.** The rename call sat *after* the `try` block, uncovered
by the same cleanup. `os.replace()` itself can genuinely raise — a
real, not hypothetical, case on Windows, where `os.replace` maps to
`MoveFileEx`/`MOVEFILE_REPLACE_EXISTING` and fails with a sharing-
violation `PermissionError` when the destination is open or locked by
another process (antivirus/backup software holding a handle, another
reader mid-`open()`) — leaking the temp file exactly like the
already-fixed case, just reached through a different trigger the first
fix didn't cover. Confirmed live: monkeypatching `os.replace` to raise
*after* `write_fn` had already produced valid content at the temp path
left that fully-written file on disk, permanently, while the original
target was correctly untouched — a leak, not data loss, but the
identical unbounded-accumulation-under-repeated-failure concern the
first fix already named. Fixed by widening the `try` block to cover
`os.replace()` too, so a failure at either step triggers the identical
cleanup, in both files.

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
other silently discarded, no error to either client.

**First fixed with a per-session, in-process `asyncio.Lock`, then
upgraded to a real cross-process one once the narrower fix's own
honestly-stated gap turned out to matter.** The first version held an
`asyncio.Lock` for the whole load-through-save span, closing the
same-process case (two browser tabs, or any two connections to one
running `sarva serve`) — but explicitly documented, not silently
assumed away, that a CLI process and a running server (or two CLI
invocations) writing the same session file was the identical bug shape
through a structurally different, cross-process channel that fix
didn't reach. Confirmed live before upgrading: two genuine
`subprocess.Popen` OS processes (not threads, not asyncio tasks — real
separate processes) doing load→sleep→save on the same session lost one
side's turn every time.

`SessionStore` now exposes `locked(name)`, a real cross-process
exclusive lock (`flock` on POSIX, `msvcrt.locking` on Windows) on a
dedicated sibling `.lock` file — the identical mechanism `sarva.
config`'s own `_exclusive_lock` already uses, applied to sessions for
the first time. Both `/chat` and `/ws/chat`, and now `sarva chat`/
`sarva run` too, wrap their entire load-through-save turn in `async
with store.locked(session):` — the CLI and the server (and two CLI
invocations, and two server connections) all now serialize against
each other through the *same* lock file, closing the whole bug class
in one mechanism rather than two separate ones. A single cross-process
lock replaced the earlier in-process-only `asyncio.Lock` entirely
(rather than keeping both), since POSIX `flock`/Windows `msvcrt.
locking` correctly serialize same-process callers against each other
exactly as well as different-process ones — no meaningful correctness
gap traded away by simplifying to one mechanism.

The blocking acquire call runs via `asyncio.to_thread`, not directly on
the event loop: acquiring this lock can genuinely block for as long as
another process's real agent turn takes, and calling a blocking
syscall directly from async code for that long would freeze every
*other* unrelated request this process is serving, not just the one
waiting on this session — the lock is held afterward simply by keeping
the file object open for the `async with` block's duration, so the
thread is only needed for the acquire/release syscalls themselves.
`locked(None)` is a deliberate no-op: a session-less turn has nothing
to protect, and locking on a shared "no session" key would needlessly
serialize every anonymous request against every other one.

**Wiring this into the CLI needed one careful adjustment, not just a
mechanical wrap:** `store.locked(session)` validates the session name
as part of entering the `async with` block, which happens *before* any
code inside that block runs — including the existing `try`/`except
ValueError` that used to be the only thing catching an invalid
`--session` name. Wrapping the load-through-save span in the lock
without also moving that catch outward would have reintroduced the raw
traceback on a bad session name the original fix (see above) closed.
Both `sarva chat` and `sarva run` now catch `ValueError` around the
whole `async with store.locked(...)` statement, not just around the
old `store.load()` call nested inside it — confirmed by the exact test
suite regression this mistake produced the first time through: three
server tests covering invalid/non-string session names failed with a
raw, uncaught `TypeError`/`ValueError` before this adjustment, caught
and fixed in the same pass before shipping.

Verified two ways: a real `httpx.AsyncClient` over the app's own ASGI
transport (not the synchronous `TestClient` used elsewhere, which
can't genuinely interleave two in-flight requests) firing two truly
concurrent `POST /chat` calls at the same session, and a genuine
two-`subprocess.Popen` test proving the cross-process case directly —
reverting `SessionStore.locked` reproduced `AttributeError: 'SessionStore'
object has no attribute 'locked'` in both new session-store tests and
every dependent server/CLI test before re-applying.

**`locked()` itself inherited a real Windows-specific bug from the
`sarva.config` lock it mirrored, found and fixed by a later sweep
re-examining the mechanism it had just copied.** The marker byte
`msvcrt.locking()` needs to exist was rewritten on every single
acquisition (a truncating `open(path, "wb")`), which is harmless for
POSIX's purely advisory `flock()` but conflicts with Windows'
*mandatory* byte-range lock: a second caller's rewrite targets exactly
the byte a first caller already holds locked, failing before the
second caller ever reaches its own lock attempt. See the packaging
chapter for the full mechanism and the `sarva.config` fix this mirrors
— fixed identically here with `os.open(..., O_CREAT)`, writing the
marker byte only once.

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

### The tokenizer was ASCII-only, making any non-ASCII memory structurally unreachable — found by finally giving this module its own dedicated fresh-eyes round

This module went untouched by more than a dozen prior rounds of
fresh-eyes sweeps — only ever brushed tangentially once, in a round
that was really about the lazy-singleton race in the *callers*
(`RememberTool`/`RecallMemoryTool`, see below) — until a round finally
gave `vector.py` itself the dedicated read the rest of the codebase had
already had many times over. `_tokenize`'s pattern, `[a-z0-9]+`, is
ASCII-only.

Confirmed live, two distinct failure modes. A pure-CJK memory (e.g.
Japanese, no ASCII characters at all) tokenized to an empty list —
giving it a zero-norm TF-IDF vector that `_cosine_similarity`
short-circuits to `0.0` for *every* query, including its own verbatim
text used as the query, tied indistinguishably with completely
unrelated English memories. That's not a ranking quirk; it means the
memory can never be surfaced by `search()`/`RecallMemoryTool` at all,
regardless of relevance — this section's own opening line promises
"relevance ranking must actually reflect real topical similarity, not
just return something," and for non-ASCII text that promise was
structurally impossible to keep. An accented Latin word fared little
better: `"café"` silently truncated to `"caf"` at the accented `é`,
matching neither `"cafe"` nor `"café"` reliably. Sarva has no
English-only restriction anywhere in its design, so a memory stored in
Japanese, Chinese, Korean, Arabic, Russian, or accented Spanish/French
is ordinary international usage, not a contrived input.

Fixed by widening the pattern to `\w+` — Unicode-aware by default for a
`str` pattern in Python 3 (no `re.ASCII` flag set), so it covers any
script with real word characters. This is an honest, not a complete,
fix: CJK scripts have no inter-word spacing, so a whole CJK sentence
still tokenizes as one long token rather than genuine per-word
segmentation — real segmentation needs a dedicated library (MeCab,
jieba), a separate, much larger feature deliberately left out of scope
here. What the fix closes is the difference between zero tokens (total
unreachability) and at least one real token (an exact or overlapping
CJK memory can now actually be found) — the specific property this
section's opening promise depends on. Verified live: the same repro's
exact-text CJK query now scores `1.0` and ranks first, instead of
`0.0` tied with unrelated memories. Verified by reverting and watching
both new tests fail with the literal old bug's own shape: `"café"`
tokenizing to `["caf"]`, and the CJK exact-match query scoring `0.0`.
2 new tests, 788 → 790 Python tests.

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

### `recall_memory`'s own `top_k` schema was purely descriptive, never enforced — a fresh-eyes sweep found the gap between what the schema promises and what actually reaches the store

`RecallMemoryTool.spec.input_schema` declares `top_k` as `{"type":
"integer"}`, but nothing between a real model's tool call and
`VectorMemoryStore.search()` ever validates that against the arguments
a model actually sends back — `input_schema` is purely descriptive,
sent to the provider so the model knows the expected shape, never
enforced server-side. Confirmed live, two distinct failure modes: a
model emitting `top_k` as the JSON string `"3"` (a real, plausible
model mistake, not a contrived attack) reached `search()`'s unguarded
`scored[:top_k]` and raised a raw, non-actionable `TypeError: slice
indices must be integers or None or have an __index__ method` —
`AgentLoop`'s own broad exception handling around tool dispatch
already keeps this from crashing the whole run, but the model sees a
bare Python exception string instead of a clean, actionable error.
Worse, silently: the schema has no `minimum` either, so a *negative*
`top_k` doesn't error at all — Python's own slice semantics turn
`top_k=-1` into "drop the last result," so a store holding exactly one
genuinely relevant memory returned `[]`, and the tool reported "No
relevant memories found" while a relevant memory actually existed — a
silently wrong answer, not just an ugly one. Fixed by validating
`top_k` explicitly in `RecallMemoryTool.run()` — the one place both
failure modes are still cheap to reject — returning the same clean,
actionable `ToolResultBlock(is_error=True, ...)` shape this file's
other tools already use for expected validation failures (an unknown
tool, a declined confirmation), rather than a raw exception or a
silently wrong empty result. `bool` is deliberately rejected too, not
silently coerced to `0`/`1` — Python's `isinstance(True, int)` is
`True`, so a JSON `top_k: true` would otherwise sail through
un-flagged. Verified live across five cases together: the string case,
the negative case, an explicit `True`, an ordinary valid integer, and
the omitted-argument default — only the first three are rejected, the
last two behave exactly as before. Verified by reverting and watching
both new tests fail with the literal old bug reproducing itself — the
raw `TypeError` for the first, the false "No relevant memories found"
for the second. 2 new tests, 741 → 743 total.

## Long-term memory: plain markdown files, one per topic

The design doc's own literal promise (§3.4): "long-term memory as plain
markdown files (human-readable, greppable)." A third memory tier,
built once the "keep pushing until every named feature is actually
built" pass reached it — genuinely distinct from the two above, not a
restyled version of either:

|  | `SessionStore` | `VectorMemoryStore` | `LongTermMemoryStore` |
|---|---|---|---|
| Scope | one conversation | usually one session | every conversation |
| Organized by | session name | session id | topic |
| Format | JSON | SQLite | plain markdown |
| Search | none (full replay) | semantic (TF-IDF cosine) | exact substring |

`sarva.memory.longterm.LongTermMemoryStore` writes one real `.md` file
per topic under `~/.sarva/memory/` (default), each a genuinely
appendable, human-readable document — open `~/.sarva/memory/
project-status.md` in any text editor and every note is right there in
plain text, headed by a UTC timestamp, no query tool required. A topic
name is slugified (`"Project Status"` and `"project-status"` land in
the same file, deliberately — the more useful behavior for a
human-organized note system, not a bug), and an unusable name (no
alphanumeric content at all) is rejected with a clean
`LongTermMemoryError` rather than producing an empty or malformed
filename.

**Search is deliberately exact substring matching, not semantic** —
`VectorMemoryStore` already covers semantic recall; the whole point of
this tier is that it's plain, greppable text, so its own search matches
that promise directly instead of duplicating the other tier's ranking.

### A real lost-update race, caught and closed before it ever shipped, not found in production later

Writing a note is a read-modify-write (read the topic's current
content, append a new entry, write the whole file back). Left
unlocked, this is the *exact* bug shape already found and fixed twice
in this codebase — `sarva.config`'s save/unset race and
`SessionStore`'s cross-process race — and reintroducing it in brand-new
code would have been a regression, not an oversight. `sarva.file_lock`
was extracted as a small, shared module (the same cross-process
`flock`/`msvcrt.locking` mechanism, now with three independent
callers instead of two hand-mirrored copies) and `LongTermMemoryStore.
write()` holds a real per-topic lock for its whole read-modify-write
span. Confirmed live, not just reasoned through: temporarily removing
the lock and racing 8 real OS threads writing to the same topic lost 7
of 8 notes to the exact race being guarded against; restoring the lock
made all 8 survive, every time.

### Wired into the agent as two more built-in tools

`NoteTool` (`note`) and `SearchNotesTool` (`search_notes`) join
`BUILTIN_TOOLS`, following the same lazy-store-construction discipline
`RememberTool`/`RecallMemoryTool` already established (opened on first
real use, not at module-import time). Deliberately NOT session-scoped
— unlike `remember`/`recall_memory`, a note written from one
conversation must be visible to every future one, which is the entire
reason this tier exists; verified directly with two different
`ToolContext`s carrying different `session_id`s, confirming a note
written under one is found by a search under the other.

### A real bug found by a later fresh-eyes sweep: an overlong topic name leaked a raw OS error, path and all

`_slugify()` had no length cap on the resulting filename. Confirmed
live: a 500-character topic name produced a slug long enough that the
filesystem itself rejected it —
`LongTermMemoryStore.write()` raised a raw `OSError` (carrying a real
local filesystem path in its own message) instead of this module's
documented `LongTermMemoryError`, and `NoteTool.run()` only ever
catches the latter, so the raw OS error surfaced straight through to
the tool result text a model or user actually sees. Fixed with
`_MAX_TOPIC_SLUG_LENGTH` (200 characters — safely under every
mainstream filesystem's ~255-byte filename-component limit, with room
for the `.md`/`.md.lock` suffix), raising the same clean, documented
`LongTermMemoryError` every other invalid-topic case already does.
Verified live that the identical 500-character topic now produces a
clean validation error with no local path anywhere in it.

### A much later fresh-eyes sweep: two differently-phrased topics that share a slug silently merged, with no record of which topic string a given entry actually used

`_slugify()` is a lossy, many-to-one normalization (lowercased,
punctuation/whitespace collapsed to `-`) with no collision check
against the file's own original topic string. Two different topic
strings that happen to slugify identically — differing only in case,
spacing, or punctuation — silently share one file, and since `write()`
only sets the top-level `# {topic}` heading when the file doesn't
exist yet, every later write under a *differently-phrased* topic
string gets filed under whichever heading the *first* write happened
to create, with nothing anywhere recording which literal string that
later write actually used. Confirmed live:
`write("Q3 Planning", "Revenue targets for Q3.")` then
`write("q3-planning", "Unrelated: my favorite pizza topping is
mushroom.")` landed both entries in one file — `list_topics()` only
ever shows `q3-planning`, and the pizza-topping note reads as
permanently filed under a "Q3 Planning" heading with no trace it came
from a differently-named call. Reachable via any real agent turn: this
tier is explicitly cross-session by design (a note written in one
conversation is visible to every future one), so the same model
producing "Meeting Notes" in one conversation and "meeting-notes" or
"Meeting notes!" in a later, unrelated one is an entirely ordinary,
non-adversarial thing to happen, not a contrived attack.

This project's own existing test for this exact collision,
`test_two_different_topic_names_that_slugify_the_same_share_one_file`,
already documents the *merge itself* as intentional, not a bug — a
human-friendly slug as file identity is meant to keep near-duplicate
topic strings from fragmenting a note across several near-identical
files, which is more useful than it is harmful. So the fix here is
narrow on purpose: only the *silence* is closed, not the merge.
Rejecting a write whose topic string doesn't exactly match the file's
original heading (the more literal "reject, don't guess" analog
`SessionStore._sanitize()` already applies to a related collision
shape) would add real friction to the common, harmless case of a
model rephrasing the same topic slightly across calls — a worse
tradeoff than the bug it would close. Fixed instead by giving every
entry its own record of the literal topic string it was actually
written under, whenever that differs from the file's original
heading: `## {timestamp} (topic: "{actual topic string}")` instead of
the bare `## {timestamp}` a same-topic repeat write still gets. The
file stays a complete, honest record of what was actually written and
under what name, instead of quietly misattributing content to
whichever topic string happened to create the file first. Verified
live with three writes together (two under "Q3 Planning," one under
"q3-planning" in between) — the differently-phrased entry alone
carries the annotation, the two same-topic entries stay unmarked, both
proven directly against the actual file content. Verified by reverting
and watching the new test fail with the literal old bug reproducing
itself — the annotation missing entirely, both entries indistinguishable
in the raw file text. 1 new test, 740 → 741 total.

### A real bug found by a fresh-eyes sweep of a genuinely different area: `note` could freeze the whole server, not just one request

After several rounds focused on the agent loop's own subagent
concurrency, a round deliberately swept a different area instead —
long-term memory. `LongTermMemoryStore.write()` is a fully synchronous
method that blocks on a real cross-process `flock` for as long as
another writer holds it (see the lost-update-race fix above), and
`NoteTool.run()` called it directly from its `async def` body with no
`asyncio.to_thread` — exactly the mistake `SessionStore.locked`'s own
fix (see the session-persistence section above) already closed once
for the identical flock-blocking shape, never propagated to this newer
tier. Confirmed live: a real second OS process holding a topic's
`.md.lock` for 3 seconds froze the *entire* event loop for the whole
window — a heartbeat coroutine that should tick roughly every 0.05s
recorded **zero** ticks across the full ~2.7s call. In a real `sarva
serve` deployment this means every other in-flight `/chat`/`/ws/chat`
turn freezes too, not just the one `note` call — the same severity
class as the session-locking bug this tier's own docstring already
names as the pattern to avoid regressing. Fixed by wrapping the call in
`asyncio.to_thread`, moving the blocking acquire (and the write itself)
off the event loop. `SearchNotesTool` only ever reads and never
contends on this lock, so it's unaffected — reasoning that turned out
to be half right, see below. Verified live the fix brings the
heartbeat count back up to 52 of ~54 expected ticks across the same
contended window. Verified by reverting and watching the new test fail
with the literal old bug's own number — `0` ticks — reproducing itself
in the assertion failure. 1 new test, 698 total.

### The "unaffected" reasoning above conflated "no lock contention" with "no blocking I/O" — `search_notes` had the identical freeze from a completely different cause

A much later fresh-eyes sweep went back to this exact file and found
the fix above's own reasoning had a gap: "`SearchNotesTool` only ever
reads and never contends on this lock, so it's unaffected" is true
about the lock specifically, but `LongTermMemoryStore.search()` is
*also* a fully synchronous method — it globs every `*.md` file under
the notes directory and calls `path.read_text()` on each one in a
loop — and `SearchNotesTool.run()` called it directly with no
`asyncio.to_thread` either, freezing the event loop the identical way
for a completely different reason (raw file I/O volume, not lock
contention). Confirmed live: 20,000 short notes (~1.8MB total, a
plausible amount after long-term real use of the `note` tool across
many conversations) froze the *entire* event loop for the whole
~360ms search — the same heartbeat methodology as the fix above, zero
ticks across the full window instead of the ~7 expected. Fixed the
same way, `asyncio.to_thread` wrapping the `search()` call. Verified
live the fix brings the heartbeat count back up to 7 of 7 expected
ticks. Verified by reverting and watching the new test fail with the
literal old bug's own number — `0` ticks. 1 new test, 729 → 730 total.
The lesson worth naming: "doesn't contend on THIS specific lock" is a
narrower claim than "doesn't block the event loop," and a comment
reasoning about the former can accidentally read as covering the
latter too if the two aren't kept explicitly separate — worth a second
look any time a tool's `async def run()` calls a method whose own
implementation was never itself audited for blocking I/O.

### The identical shape, one round later, in `remember`/`recall_memory` — and a second bug hiding behind the first

The very next round applied the same lens to the other two memory
tools sharing this file's chapter: `remember`/`recall_memory`
(`sarva.memory.vector`). Python's `sqlite3` module blocks for up to its
own default 5-second `timeout` waiting for another connection's write
lock to clear before raising "database is locked," and `RememberTool.
run()`/`RecallMemoryTool.run()` both called `add()`/`search()` directly
with no `asyncio.to_thread` — the identical mistake `NoteTool` had just
been fixed for, in a sibling tool sharing this exact chapter. Confirmed
live: a real second OS process holding a genuine SQLite `BEGIN
EXCLUSIVE` lock on the same database for 3 seconds froze the calling
process's entire event loop for the whole window (0 of ~61 expected
heartbeat ticks).

**A second, genuinely separate bug was hiding behind the first, caught
mid-fix rather than found by a fresh sweep of its own:** both tools'
own default store is opened lazily on first `run()` (`_get_store()`),
and `VectorMemoryStore.__init__` does its own `CREATE TABLE IF NOT
EXISTS` + `commit()` against the same database — just as capable of
blocking on the identical contended lock as `add()`/`search()`
themselves. Wrapping only the already-open store's method call in
`asyncio.to_thread` (mirroring `NoteTool`'s own fix exactly) would have
left the freeze fully reachable on a tool's very first call in a fresh
process — caught by the same live repro, re-run after the first
attempt, showing the identical zero-tick freeze despite the "fix"
already being in place. Closed by folding the lazy construction and
the method call into one `asyncio.to_thread` dispatch (`RememberTool.
_add`/`RecallMemoryTool._search`) so both blocking-prone operations
move off the event loop together.

Fixing this also surfaced a real thread-affinity constraint `NoteTool`'s
own fix never had to deal with: unlike `LongTermMemoryStore.write()`
(self-contained, opens a fresh lock file per call), `VectorMemoryStore`
holds one persistent `sqlite3.Connection` for its whole lifetime, and
Python's `sqlite3` module by default only permits a connection to be
used from the exact thread that created it — confirmed live, a naive
`asyncio.to_thread(conn.execute, ...)` against a connection created on
the event-loop thread raised `ProgrammingError: SQLite objects created
in a thread can only be used in that same thread`, since `asyncio.
to_thread` dispatches to a pool thread that can differ from call to
call. Fixed by connecting with `check_same_thread=False` and adding an
explicit `threading.Lock` around every real use of the connection —
disabling sqlite3's own safety check alone doesn't make one connection
safe for genuinely concurrent access from multiple threads, so the
lock does the actual serializing.

Verified live the fix brings the heartbeat count back up to 58 of ~60
expected ticks across the identical contended window. Verified by
reverting and watching the new test fail with the literal old bug's
own number — `0` ticks — reproducing itself. 1 new test, 698 -> 699
total.

### The lazy-construction fix caught above, for `remember`/`recall_memory`, never made its way back to `note` — the tool that originally inspired this whole chapter

A much later fresh-eyes sweep, comparing `NoteTool` against its own
sibling `RememberTool`, found the gap directly: the "second, genuinely
separate bug" fixed above — folding a tool's lazy `_get_store()`
construction together with its store call into one `asyncio.to_thread`
dispatch, since the construction itself does real blocking I/O — was
discovered and fixed for `RememberTool`/`RecallMemoryTool` in the very
round *after* `NoteTool`'s own flock-contention fix shipped, but never
propagated backward into `NoteTool` itself, even though it's the tool
whose original fix this whole chapter is built around.
`NoteTool.run()` still called `self._get_store()` as a plain argument
expression, evaluated eagerly on the event loop *before*
`asyncio.to_thread` ever got control — only the subsequent `.write`
call was actually dispatched to a thread. `LongTermMemoryStore.
__init__` does real, blocking filesystem I/O (`Path.mkdir` + `os.
chmod`), which can be slow on a contended or network-mounted
filesystem — this project's own `sarva.config` docstring names "shared
dev servers, lab machines, CI runners with persistent home
directories" as a real, not hypothetical, scenario. Confirmed live: a
deliberately slowed `LongTermMemoryStore.__init__` (simulating exactly
that) froze the event loop for the whole ~1s construction — a
heartbeat coroutine that should tick roughly every 0.05s recorded
essentially zero ticks. The existing contended-lock test for this tool
passes a pre-built `store=store`, bypassing `_get_store()`'s lazy
construction entirely and only exercising `write()`'s own lock
contention — exactly why this gap slipped past it undetected the whole
time. Fixed identically to `RememberTool`'s own shape: a `_write`
helper wraps both the lazy construction and the `write()` call
together, dispatched as one unit through `asyncio.to_thread`. Verified
live the fix brings the heartbeat count back up to 20 of 20 expected
ticks across the same slowed-construction window. Verified by
reverting and watching the new test fail with the literal old bug's
own near-zero number reproducing itself. 1 new test, 738 -> 739 total.

### The fourth sibling had it too — `search_notes` never got the lazy-construction fix either, just one tool later in the same file

The very next round applied the exact same comparison one tool
further: `SearchNotesTool` sits right below `NoteTool` in this file,
and its own `run()` still called `self._get_store()` as a plain
argument expression outside the `asyncio.to_thread` dispatch — the
identical gap `NoteTool` had just been fixed for, in the tool
immediately preceding it. `SearchNotesTool`'s own existing comment (see
the earlier `search_notes` fix above) correctly reasoned through
`search()` itself needing `asyncio.to_thread`, but stopped there
without noticing `_get_store()`'s own lazy construction needed to be
*inside* that dispatched call too. Confirmed live with the identical
methodology: a deliberately slowed `LongTermMemoryStore.__init__`
froze the event loop for the whole ~1s construction on `search_notes`'s
first real call, near-zero heartbeat ticks recorded. The existing
large-notes-directory test for this tool passes a pre-built `store=
store`, the same masking shape that let the `NoteTool` gap slip past
its own contended-lock test undetected. Fixed identically: a `_search`
helper wraps both the lazy construction and the `search()` call
together, dispatched as one unit through `asyncio.to_thread`. Verified
live the fix brings the heartbeat count back up to 20 of 20 expected
ticks. Verified by reverting and watching the new test fail with the
literal old bug's own near-zero number reproducing itself. With this
fix, all four memory tools in this file (`remember`, `recall_memory`,
`note`, `search_notes`) now dispatch their entire lazy-construction-
plus-store-call path through `asyncio.to_thread` as a single unit,
closing this exact bug shape across the whole chapter, not just three
of its four tools. 1 new test, 739 -> 740 total.

### Dispatching that lazy construction onto real OS threads made a completely different bug live for the first time — an unsynchronized singleton race, in all four tools at once

Fixing the event-loop-freeze shape above (routing `_get_store()`'s lazy
construction through `asyncio.to_thread` in all four tools) had a real,
un-anticipated side effect a much later fresh-eyes sweep caught: it
made the classic unsynchronized check-then-act singleton race genuinely
*live* for the first time. `_get_store()`'s `if self._store is None:
self._store = ...` now runs on a real OS worker thread, not a
cooperative asyncio task — and `BUILTIN_TOOLS` is a module-level
singleton list, the same `RememberTool()`/`RecallMemoryTool()`/
`NoteTool()`/`SearchNotesTool()` instances handed to *every* `AgentLoop`
a running `sarva serve` process builds, one per `/ws/chat` connection.
Two concurrent connections — two users, or two tabs/windows of the
same user — both calling the same tool before its very first call
(most realistically right after server startup) can both pass
`self._store is None` simultaneously and each construct their own
independent store object: its own SQLite connection and lock for the
two memory tools, its own directory handle for the two notes tools.
Whichever assignment lands last silently wins; the other is discarded
without ever being closed, and — worse than a mere resource leak — the
thread that "lost" the race ends up operating through a store object
it didn't itself construct, momentarily breaking the exact "one lock
serializes every real access" invariant `VectorMemoryStore`'s own
docstring claims, for the duration of the race.

Confirmed live with a deterministic repro (a small artificial delay
inside the relevant store's `__init__`, the same technique the freeze
tests above already use): two concurrent `remember` calls on one
shared `RememberTool` instance — the exact shape `BUILTIN_TOOLS`
produces — reliably constructed two distinct `VectorMemoryStore`
objects for the one lazily-cached `self._store` field, 10/10 trials.
Fixed with the standard double-checked-locking pattern: a
`threading.Lock` (not `asyncio.Lock`, which isn't safe to hand between
real OS threads) created once per tool instance in `__init__`, guarding
the check-and-construct in all four tools identically — the lock is
only ever contended during this narrow first-use window, never on the
hot path once `self._store` is set. Honest about severity: the blast
radius is a momentary resource blip during the race window, not
lasting data corruption — both racing connections still ultimately
write to the same on-disk file, and the situation self-heals the
instant the race resolves. Verified live the identical repro now
constructs exactly one store object across 10/10 trials. Verified by
reverting and watching the new tests fail with the literal old bug's
own shape: two distinct store objects constructed for one field. New
tests cover both underlying store types this fix touches
(`VectorMemoryStore` via `remember`, `LongTermMemoryStore` via `note`)
— the other two tools share the identical fix and mechanism. 2 new
tests, 773 → 775 Python tests.

### A file that exists but is empty crashed `write()` with a raw `IndexError`, silently losing the note — only "doesn't exist yet" had ever been special-cased

A much later fresh-eyes sweep found a gap in `write()`'s own
read-existing-content step: `path.is_file()` only distinguishes "no
file yet" from "a file is there" — it says nothing about whether that
file actually has any content. `existing = path.read_text(...) if
file_exists else f"# {stripped_topic}\n"` special-cases the former but
not a real third state: a topic `.md` file that exists on disk but is
0 bytes. `"".splitlines()` returns `[]`, not `[""]`, so the very next
line, `existing.splitlines()[0].removeprefix("#").strip()`, raised a
raw, uncaught `IndexError` — before `atomic_write_text` was ever
reached, so the new note was silently lost, not just delayed.

This isn't a contrived state to reach: this store's own docstring
calls its files "human-readable ... a person can open in any editor
and read or hand-edit directly" — a user (or any external process)
clearing a note file's contents is ordinary use of the exact feature
this tier is designed around, not adversarial input. `NoteTool.run()`
only catches `LongTermMemoryError`, so this bug leaked straight past
the "reject, don't guess" discipline every sibling case in `write()`
already follows correctly — saved from crashing a live `sarva serve`
process only by `AgentLoop.run_one()`'s own incidental broad `except
Exception` one layer up, a safety net no other/future direct caller of
this store would have.

Confirmed live: a note written, then its file truncated to 0 bytes,
then written to again, reliably raised `IndexError: list index out of
range` inside `write()`. Fixed by treating an existing-but-blank file
the same as "doesn't exist yet" for heading purposes — there's no real
original topic left to compare against or preserve, so the file is
reseeded with a fresh `# Topic` heading exactly like a brand-new file
gets, rather than crashing. Verified live the identical repro now
succeeds, producing a correctly re-seeded file. Verified by reverting
and watching the new test fail with the literal old bug's own shape:
`IndexError: list index out of range` at the same line. 1 new test,
816 → 817 Python tests.

### `_slugify` had the identical ASCII-only pattern already found and fixed for `vector.py`'s tokenizer — never propagated to this sibling module

A much later fresh-eyes sweep, applying a "structural sibling
comparison" lens across the three memory-tier modules this chapter
opens by naming, found that `_slugify`'s own normalization pattern,
`[^a-z0-9]+`, had the identical ASCII-only shape already found and
fixed once for `sarva.memory.vector`'s `_tokenize()` (the chapter
above) — the fix never propagated one module over, to this sibling
doing the same normalization job for a different purpose (a
filesystem-safe slug instead of a search token).

A topic written entirely in a non-Latin script — Japanese, Russian,
Arabic, or any other script with no ASCII alphanumeric characters at
all, an entirely ordinary thing for a non-English-speaking user or a
model conversing in another language to ask this tool to remember
something under — slugified to an empty string and was rejected
outright with `LongTermMemoryError`, making the `note` tool completely
unusable for that topic. An accented Latin topic like `"café"` wasn't
rejected but was silently mangled to `"caf"`, the identical truncation
already fixed for `vector.py`. Confirmed live before this fix:
`_slugify("日本語のメモ")` raised `LongTermMemoryError`; `_slugify("café")`
returned `"caf"`.

Fixed by widening the pattern to `[\W_]+` — Unicode-aware `\W` by
default for a `str` pattern in Python 3, matching `vector.py`'s own
fix, with `_` explicitly still collapsed to `-` too so this function's
own prior treatment of underscores (`\w` alone would keep them as a
"word" character, a silent behavior change this fix deliberately
avoided). Widening the character set that survives slugifying also
reopened a second, related gap in `_MAX_TOPIC_SLUG_LENGTH`'s own
check, found in the same pass: that cap was measured in Python
characters, exactly equivalent to bytes back when the slug was
ASCII-only, but a non-Latin character can take up to 4 bytes in UTF-8
— the actual unit the filesystem's own filename-length limit this cap
exists to protect against is measured in — so a slug well under the
200-*character* cap could still exceed the real 255-*byte* filesystem
limit, reintroducing the exact raw-`OSError` bug this cap was built to
prevent (see the earlier fix above), just for Unicode topics instead
of long ASCII ones. Fixed by measuring `len(slug.encode("utf-8"))`
instead of `len(slug)`. Verified live both fixes hold: non-Latin and
accented topics now slugify successfully and correctly; 100 CJK
characters (300 UTF-8 bytes, comfortably under the old 200-*character*
cap) is correctly rejected under the new byte-aware cap. Verified by
reverting and watching all three new tests fail with the literal old
bug's own shape: `LongTermMemoryError` for a pure-non-Latin topic,
`"caf"` for an accented one. 3 new tests, 844 → 847 Python tests.

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
