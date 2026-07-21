"""sarva.memory — persistent state across sessions.

Deliberately simple: plain files, human-readable, greppable — the same
philosophy as Claude Code's own memory files (design doc §3.4). A vector
index or database-backed store can layer on top later without changing
this contract.
"""
