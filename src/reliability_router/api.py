"""FastAPI application."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .models import RouterRequest, RouterResponse, RoutingDecision
from .service import ReliabilityRouter, get_default_router


class BatchRouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: list[RouterRequest] = Field(min_length=1, max_length=100)


class BatchRouteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[RoutingDecision]


app = FastAPI(
    title="Reliability Router",
    version="1.0.0",
    description="Risk-aware QUICK/DEEP routing with evidence checks and fail-closed post-verification.",
)

RouterDep = Annotated[ReliabilityRouter, Depends(get_default_router)]


@app.get("/healthz")
def health(router: RouterDep) -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": getattr(router.provider, "name", "unknown"),
        "policy_version": router.config.policy_version,
    }


@app.get("/v1/policy")
def policy(router: RouterDep) -> dict[str, Any]:
    return router.policy_snapshot()


@app.post("/v1/route", response_model=RoutingDecision)
def route(request: RouterRequest, router: RouterDep) -> RoutingDecision:
    try:
        return router.route(request)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=f"Routing failed: {exc}") from exc


@app.post("/v1/route:batch", response_model=BatchRouteResponse)
def batch_route(request: BatchRouteRequest, router: RouterDep) -> BatchRouteResponse:
    return BatchRouteResponse(decisions=[router.route(item) for item in request.requests])


@app.post("/v1/answer", response_model=RouterResponse)
def answer(request: RouterRequest, router: RouterDep) -> RouterResponse:
    try:
        return router.answer(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc
