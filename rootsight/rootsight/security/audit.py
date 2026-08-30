"""Append-only audit log.

Every access decision, every denial, every model call and every narrative
release writes one immutable event.  The log is the artefact an auditor reads;
it is deliberately boring and deliberately complete.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

from .. import config

_LOCK = threading.Lock()
LOG_PATH = os.path.join(config.ARTIFACT_DIR, "audit_log.jsonl")


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    recorded_at: str
    event_type: str
    actor: str
    role: str
    resource: str
    outcome: str            # ALLOW | DENY | RELEASE | DOWNGRADE
    detail: dict = field(default_factory=dict)
    request_id: str | None = None


class AuditLog:
    def __init__(self, path: str = LOG_PATH, persist: bool = True):
        self.path = path
        self.persist = persist
        self.events: list[AuditEvent] = []
        if persist:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, *, event_type: str, actor: str, role: str, resource: str,
               outcome: str, detail: dict | None = None,
               request_id: str | None = None) -> AuditEvent:
        ev = AuditEvent(
            event_id=f"aud-{uuid.uuid4().hex[:12]}",
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            event_type=event_type, actor=actor, role=role, resource=resource,
            outcome=outcome, detail=detail or {}, request_id=request_id)
        self.events.append(ev)
        if self.persist:
            with _LOCK, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(ev)) + "\n")
        return ev

    def for_request(self, request_id: str) -> list[dict]:
        return [asdict(e) for e in self.events if e.request_id == request_id]

    def denials(self) -> list[dict]:
        return [asdict(e) for e in self.events if e.outcome == "DENY"]

    def tail(self, n: int = 50) -> list[dict]:
        return [asdict(e) for e in self.events[-n:]]


_LOG: AuditLog | None = None


def audit_log() -> AuditLog:
    global _LOG
    if _LOG is None:
        _LOG = AuditLog()
    return _LOG
