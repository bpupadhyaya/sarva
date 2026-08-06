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

**A second real bug, a genuine memory-exhaustion DoS, found by actually
generating a real (non-corrupted, non-malicious) video and measuring
peak memory:** the decode used to be `frames = list(container.decode(
stream))` -- every frame of the ENTIRE video decoded and held in memory
at once, before sampling just `_MAX_SAMPLED_FRAMES` from that fully-
materialized list. Confirmed live: a completely ordinary 60-second,
1280x720, 30fps synthetic MP4 (834 KB compressed, generated with
plain `ffmpeg`, no fuzzing/corruption at all) drove peak RSS to 2.6 GB
-- roughly 3000x the file's own size. A longer or higher-resolution
attachment (a 20-minute screen recording is an entirely ordinary thing
for a real user to attach) scales this linearly toward tens of GB,
exhausting host memory well before the 30s decode timeout `video.py`
already enforces even fires -- that timeout bounds wall-clock time, not
memory, so it does nothing to stop this. Unlike the SIGBUS bug above,
no corrupted/crafted file is needed; a completely legitimate video
attachment of entirely ordinary size triggers this.

Fixed by sampling via `container.seek()` to each target timestamp
(computed from the stream's own duration, evenly spaced) and decoding
forward only until the nearest actual frame at that timestamp is found
-- never materializing more than one frame at a time, and never
decoding more of the stream than the distance between keyframes (a
video's GOP structure, typically a few seconds' worth of frames, not
the video's total length). Confirmed live: peak RSS for the identical
60-second test video dropped from 2.6 GB to ~37 MB (a ~70x reduction)
using this approach, landing on frames at the exact requested
timestamps (`0.00s`/`15.00s`/`30.00s`/`45.00s` for a 60s video sampled
4 ways), not merely close to them. A per-sample decoded-frame cap
(`_MAX_FRAMES_PER_SAMPLE`) bounds the worst case further, for a
pathological encoding with an unusually large GOP.

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

# A safety valve, not the primary fix: bounds how far forward a single
# sample's seek-then-decode is allowed to scan looking for its target
# frame, in case a pathological encoding has an unusually large gap
# between keyframes. In ordinary encodings this is never reached -- the
# real fix is seeking near the target first, so decoding only ever
# covers one GOP's worth of frames per sample, not the whole video.
_MAX_FRAMES_PER_SAMPLE = 600


def _sample_near(container, stream, target_pts: int):
    """Seeks to the keyframe at-or-before `target_pts`, then decodes
    forward, keeping only the single frame closest to `target_pts` seen
    so far -- never a list of frames -- stopping as soon as a frame at
    or past the target is found (or the per-sample scan cap is hit).
    Returns `None` if nothing decodes at all past this seek point."""
    container.seek(target_pts, stream=stream, any_frame=False, backward=True)
    best = None
    for scanned, frame in enumerate(container.decode(stream), start=1):
        if frame.pts is not None and (
            best is None or abs(frame.pts - target_pts) < abs(best.pts - target_pts)
        ):
            best = frame
        if frame.pts is not None and frame.pts >= target_pts:
            break
        if scanned >= _MAX_FRAMES_PER_SAMPLE:
            break
    return best


def _run(raw: bytes) -> int:
    try:
        with av.open(io.BytesIO(raw)) as container:
            if not container.streams.video:
                return 1
            stream = container.streams.video[0]
            duration_s = float(stream.duration * stream.time_base) if stream.duration else None

            if not duration_s or duration_s <= 0:
                # No known/positive duration -- can't compute evenly
                # spaced timestamps, so just take whatever the first
                # decoded frame is (still bounded: at most one GOP's
                # worth of frames scanned, via the same helper).
                #
                # A real bug found by a fresh-eyes sweep: this branch
                # already treats a NEGATIVE duration_s (not just None/0)
                # as "no known duration" for sampling purposes -- but the
                # write-back below only substituted the NaN sentinel when
                # duration_s was exactly None, so a negative value (a
                # real, non-null float straight from a lightly-corrupted
                # container's metadata -- exactly the "corrupted metadata,
                # still-decodable frames" threat model this whole worker
                # exists to defend against) sailed through unmodified into
                # the duration field this module's own docstring promises
                # is "NaN if no known duration," surfacing all the way to
                # the user/model as a nonsensical "-5.0s" instead of
                # "unknown duration." Normalized here so the write-back
                # matches this branch's own sampling decision exactly.
                duration_s = None
                frame = _sample_near(container, stream, 0)
                sampled = [frame] if frame is not None else []
            else:
                count = _MAX_SAMPLED_FRAMES
                sampled = []
                for i in range(count):
                    target_s = duration_s * i / count
                    target_pts = int(target_s / stream.time_base)
                    frame = _sample_near(container, stream, target_pts)
                    if frame is not None:
                        sampled.append(frame)
    except (FFmpegError, ValueError):
        return 1

    if not sampled:
        return 1

    out = sys.stdout.buffer
    out.write(struct.pack(">d", duration_s if duration_s is not None else float("nan")))
    for frame in sampled:
        buf = io.BytesIO()
        frame.to_image().save(buf, format="PNG")
        data = buf.getvalue()
        out.write(struct.pack(">I", len(data)))
        out.write(data)
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(_run(sys.stdin.buffer.read()))
