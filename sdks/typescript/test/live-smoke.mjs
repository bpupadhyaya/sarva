// Not part of the automated test suite (needs a real running `sarva
// serve` process) -- a one-off live smoke test proving this SDK genuinely
// talks to the real Python server, not just that its own unit tests
// (which mock fetch/WebSocket) pass in isolation.
import { SarvaClient } from "../dist/index.js";

const client = new SarvaClient({ baseUrl: process.argv[2] ?? "http://127.0.0.1:8123" });

const health = await client.health();
console.log("health:", health);
if (health.status !== "ok") throw new Error("health check failed");

const models = await client.models();
console.log("models:", models.map((m) => `${m.id} (${m.available ? "available" : "unavailable"})`));
if (!models.some((m) => m.id === "mock" || m.available)) {
  throw new Error("expected at least the mock model to be listed");
}

const chatResp = await client.chat({ message: "hello from the real TypeScript SDK" });
console.log("chat response:", chatResp);
if (chatResp.state !== "done" || !chatResp.message) {
  throw new Error(`expected a real done response, got ${JSON.stringify(chatResp)}`);
}

console.log("\nAll live checks passed -- the SDK genuinely talks to a real sarva serve process.");
