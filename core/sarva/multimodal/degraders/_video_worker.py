"""sarva.multimodal.degraders._video_worker — isolates PyAV's native
decoder in its own subprocess.

**A real bug found by directly fuzzing a valid MP4** (random byte flips
across many trials) and running the exact `av.open`/`container.decode`
call this project's own video degrader makes: 3 of 60 seeded trials
didn't raise a Python exception at all — they killed the process
outright with a real SIGBUS (signal 10), a native memory fault inside
PyAV/libavcodec that no `try`/`except`, however broad, can catch. A
user- or attacker-supplied corrupted video attachment could crash the
entire `sarva serve`/CLI process, not just fail one degradation
attempt — the same severity class as the earlier `AgentLoop`
degradation-fallback crash, but this one is a native fault no amount of
widening a Python `except` clause can fix.

Running the actual decode in an isolated subprocess (this module,
invoked via `python -m`) means a native crash here kills only the
subprocess — the parent (`video.py`) sees a nonzero/negative exit code
and falls back to the existing metadata-only report, exactly the same
path already used for a plain "couldn't decode this" failure. Decode
failure and native crash are deliberately indistinguishable to the
caller: both mean the same thing, "couldn't safely decode this, fall
back," and the caller doesn't need to know which one happened.

Reads raw video bytes from stdin. On success, writes a simple
length-prefixed binary framing to stdout: 8 bytes (big-endian double,
NaN if no known duration), then per sampled frame a 4-byte big-endian
length followed by that many PNG bytes. Exits nonzero (and writes
nothing to stdout) on any failure.
"""

from __future__ import annotations

import io
import struct
import sys

import av
from av.error import FFmpegError

_MAX_SAMPLED_FRAMES = 4


def _run(raw: bytes) -> int:
    try:
        with av.open(io.BytesIO(raw)) as container:
            if not container.streams.video:
                return 1
            stream = container.streams.video[0]
            duration_s = float(stream.duration * stream.time_base) if stream.duration else None
            frames = list(container.decode(stream))
    except (FFmpegError, ValueError):
        return 1

    if not frames:
        return 1

    # Evenly spaced indices across the decoded frames, capped at
    # _MAX_SAMPLED_FRAMES -- must match video.py's own constant, since
    # this worker is the only thing that actually samples frames now.
    count = min(_MAX_SAMPLED_FRAMES, len(frames))
    step = len(frames) / count
    indices = [int(i * step) for i in range(count)]

    out = sys.stdout.buffer
    out.write(struct.pack(">d", duration_s if duration_s is not None else float("nan")))
    for i in indices:
        buf = io.BytesIO()
        frames[i].to_image().save(buf, format="PNG")
        data = buf.getvalue()
        out.write(struct.pack(">I", len(data)))
        out.write(data)
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(_run(sys.stdin.buffer.read()))
