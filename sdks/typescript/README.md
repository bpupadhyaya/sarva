# sarva-sdk

A thin TypeScript client for the Sarva REST + WebSocket API
(`core/sarva/server/app.py`). Named directly in the design doc's own
repo-structure diagram (`sdks/typescript/ # thin REST/WS client`) and
tech-stack table ("Python (the core itself) + thin TypeScript client
for the REST/WS API").

Every method maps to exactly one real server endpoint. No client-side
retries, caching, or hidden state beyond what's documented below.

## Install

This package isn't published to npm yet — consume it from the repo
directly (`sdks/typescript/`), or `npm run build` and point another
package at `sdks/typescript/dist`.

## Usage

```ts
import { SarvaClient } from "sarva-sdk";

const client = new SarvaClient({ baseUrl: "http://127.0.0.1:8000" });

// REST — single-turn, no tools (mirrors `sarva chat`)
const response = await client.chat({ message: "hello" });
console.log(response.message);

// WebSocket — tool-using, streaming (mirrors `sarva run`)
const stream = client.chatStream(
  { message: "list the files here", auto: true },
  {
    onEvent: (event) => {
      if (event.type === "model_stream" && event.event.type === "text_delta") {
        process.stdout.write(event.event.text);
      }
      if (event.type === "needs_confirmation") {
        // auto: true above means nothing is waiting to consume a reply —
        // do NOT call respondToConfirmation() in that mode (see below).
      }
      if (event.type === "run_done") {
        stream.close();
      }
    },
  },
);
```

## ⚠️ Node.js: you must provide a WebSocket implementation

**Node's own built-in global `WebSocket` does not interoperate
correctly with Sarva's server.** Confirmed live, not a hypothetical:
the connection opens and the initial request frame sends successfully,
then the connection silently drops (close code 1006) before any
server response ever arrives — while the exact same request through
the well-established [`ws`](https://www.npmjs.com/package/ws) npm
package, and through Python's own `websockets` client, both work
correctly against the identical running server. This is a real,
narrow interop gap specific to Node's implementation (built on
`undici`), not a bug in this SDK or in Sarva's server — a real
standards-compliant client (`ws`, and Python's client) both prove the
server side is correct.

Because of this, `SarvaClient` deliberately does **not** default to
the global `WebSocket` when running under Node, even though one
exists as of Node 22+ — silently handing back an implementation known
not to work would violate the same "reject, don't guess" discipline
the rest of this project applies to malformed input. `chatStream()`
throws immediately with an actionable message if you're on Node and
haven't passed one explicitly:

```ts
import WebSocketImpl from "ws";

const client = new SarvaClient({
  baseUrl: "http://127.0.0.1:8000",
  webSocketImpl: WebSocketImpl,
});
```

Browsers are unaffected — their `WebSocket` implementation is a
different, long-established one — and `SarvaClient` keeps defaulting
to the global there, no configuration needed.

## `fetch`

Same shape: `SarvaClient` uses the global `fetch` by default (available
in every modern browser and Node 18+); pass `fetchImpl` explicitly only
if you're on an older Node runtime with no global `fetch`.

## Development

```bash
npm install
npm run build   # tsc -> dist/
npm test        # vitest — mocked fetch/WebSocket, no real server needed
```

`test/live-smoke.mjs` and `test/live-smoke-ws.mjs` are one-off, manual
live-verification scripts (not part of the automated suite — they need
a real running `sarva serve` process) proving this client genuinely
talks to the real Python server, not just that its own mocked unit
tests pass in isolation:

```bash
sarva serve --port 8123 &
npm run build
node test/live-smoke.mjs http://127.0.0.1:8123      # REST
node test/live-smoke-ws.mjs http://127.0.0.1:8123   # WebSocket (needs `ws`, already a devDependency)
```

## What's honestly not covered yet

- `ContentBlock` only fully types the `text` block shape (`TextBlock`);
  every other content block (image, audio, thinking, tool blocks)
  comes through as a loosely-typed `{ type: string; ... }` rather than
  a fully modeled union — the exact JSON encoding of Python's
  `bytes | None` media fields isn't pinned down here. `textFromContent()`
  covers the common case (a message's plain text) without needing the
  full union.
- No convenience wrapper around the desktop app's own React
  integration yet — `apps/desktop/src/events.ts` still hand-mirrors a
  subset of these same types locally; that file's own comment already
  names this SDK's existence as the reason to eventually consolidate,
  left as a deliberate follow-up rather than done in the same change
  that introduced this package.
