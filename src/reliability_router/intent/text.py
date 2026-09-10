"""One request -> text serializer, shared by every artifact-backed parser.

Training and runtime must produce byte-identical text for the same request. Keeping this in one
family-neutral module is what stops a parser from being trained on one rendering and served
another, which fails silently as a few points of accuracy rather than as an error.
"""

from __future__ import annotations

from ..models import RouterRequest


def request_to_classifier_text(request: RouterRequest) -> str:
    """Render a request the way every classifier family sees it."""
    context = "\n".join(
        f"[context:{index + 1} source={item.source or 'unknown'} trust={item.trust:.2f}] {item.text}"
        for index, item in enumerate(request.context)
    )
    tools = ", ".join(request.available_tools) if request.available_tools else "none"
    return f"Query:\n{request.query}\n\nAvailable tools:\n{tools}\n\nContext:\n{context or 'none'}"
