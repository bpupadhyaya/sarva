import WS from "ws";
import { SarvaClient } from "../dist/index.js";

const client = new SarvaClient({
  baseUrl: process.argv[2] ?? "http://127.0.0.1:8123",
  webSocketImpl: WS,
});

const events = [];
await new Promise((resolve, reject) => {
  const stream = client.chatStream(
    { message: "say hi", auto: true },
    {
      onEvent: (event) => {
        events.push(event);
        console.log("event:", event.type, "state" in event ? event.state : "");
        if (event.type === "run_done") {
          stream.close();
          resolve();
        }
      },
      onError: (err) => reject(err),
      onClose: () => {},
    },
  );
  setTimeout(() => reject(new Error("timed out waiting for run_done")), 10_000);
});

const runDone = events.find((e) => e.type === "run_done");
if (!runDone || runDone.state !== "done") {
  throw new Error(`expected a real done run_done event, got ${JSON.stringify(runDone)}`);
}
console.log("\nAll live WebSocket checks passed -- real streaming turn over /ws/chat, via the ws package.");
