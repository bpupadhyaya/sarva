"""sarva.server — REST + WebSocket over the agent loop.

Same skin philosophy as the CLI: this is a thin wrapper over
`sarva.agent.loop.AgentLoop` and `sarva.runtime`. No business logic lives
here that isn't already in the core engine — the server is how a future web
UI or desktop app talks to the same agent the CLI drives.
"""
