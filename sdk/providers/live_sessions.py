"""Live sessions held open between HTTP requests while the client runs tools.

An OpenAI-style agent loop is stateless: the model asks for tools, the client
runs them, then posts the whole history again. The Live API cannot take that
history back — `function_call` parts in client content are rejected with 1007 —
so every tool round used to open a fresh session with the conversation
flattened into text, and the model lost track of what it had already done.
Parking the session that issued the calls, and resuming it with
`send_tool_response` once the results arrive, keeps the model's real context.

State lives in this process. The service runs a single uvicorn worker; with
more, a result that lands on another worker finds nothing parked and falls
back to replaying the history, which is how every request behaved before.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Clients run tools locally — package installs, test suites — before posting
# the results, so a parked session has to survive minutes, not seconds.
IDLE_TIMEOUT_S = 600.0
MAX_PARKED = 64


@dataclass(frozen=True)
class PendingCall:
    name: str
    # None when Gemini sent no id and the public one was synthesised; the
    # response then goes back without an id rather than with an invented one.
    gemini_id: Optional[str]


@dataclass(eq=False)
class ParkedSession:
    connection: object  # the async context manager returned by live.connect()
    session: object
    model: str
    pending: dict[str, PendingCall]  # public call id → call
    parked_at: float = field(default_factory=time.monotonic)

    async def close(self) -> None:
        await close_connection(self.connection)


_by_call_id: dict[str, ParkedSession] = {}
_lock = asyncio.Lock()


async def close_connection(connection) -> None:
    try:
        await connection.__aexit__(None, None, None)
    except Exception as exc:  # already gone upstream; nothing left to release
        logger.debug("Closing Live session failed: %s", exc)


def _unpark(entry: ParkedSession) -> None:
    for call_id in entry.pending:
        if _by_call_id.get(call_id) is entry:
            del _by_call_id[call_id]


def _evict_locked(now: float) -> list[ParkedSession]:
    """Unpark expired sessions, then the oldest ones past the cap."""
    entries = sorted({id(e): e for e in _by_call_id.values()}.values(), key=lambda e: e.parked_at)
    evicted = [e for e in entries if now - e.parked_at > IDLE_TIMEOUT_S]
    remaining = [e for e in entries if e not in evicted]
    if len(remaining) > MAX_PARKED:
        evicted.extend(remaining[: len(remaining) - MAX_PARKED])
    for entry in evicted:
        _unpark(entry)
    return evicted


async def _close_all(entries: Iterable[ParkedSession]) -> None:
    for entry in entries:
        await entry.close()


async def park(entry: ParkedSession) -> None:
    async with _lock:
        for call_id in entry.pending:
            _by_call_id[call_id] = entry
        evicted = _evict_locked(time.monotonic())
    await _close_all(evicted)


async def claim(call_ids: Iterable[str], model: str) -> Optional[ParkedSession]:
    """Take the parked session these results answer, if it can resume.

    It must be waiting on exactly these calls, on the same model. Anything else
    means the conversation moved on without it, so it is closed instead.
    """
    ids = set(call_ids)
    async with _lock:
        evicted = _evict_locked(time.monotonic())
        owners = {id(_by_call_id[c]): _by_call_id[c] for c in ids if c in _by_call_id}
        entry = None
        for candidate in owners.values():
            _unpark(candidate)
            if len(owners) == 1 and set(candidate.pending) == ids and candidate.model == model:
                entry = candidate
            else:
                evicted.append(candidate)
    await _close_all(evicted)
    return entry


async def discard(call_ids: Iterable[str]) -> None:
    """Close the session behind these calls; nothing will answer them."""
    async with _lock:
        entries = {id(_by_call_id[c]): _by_call_id[c] for c in call_ids if c in _by_call_id}
        for entry in entries.values():
            _unpark(entry)
    await _close_all(entries.values())
