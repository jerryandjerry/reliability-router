"""Intent parser contract."""

from __future__ import annotations

from typing import Protocol

from ..models import RouterRequest, RoutingDecision


class IntentParser(Protocol):
    """Maps a request to the router's stable QUICK/DEEP decision contract."""

    name: str
    version: str

    def parse(self, request: RouterRequest) -> RoutingDecision: ...
