"""``GET /models`` — registry read-only inspection endpoint (Spec 20).

Returns the contents of the ``model_registry`` SQLite table ordered
newest-first. Useful for ops + debugging — confirms "which model is
live?" without shell access to the DB.

This route does NOT mutate the registry. Activation is a separate
admin operation that's deliberately out of scope for v1 (it requires
restarting the FastAPI process so the lifespan handler picks up the
new active row; a hot-swap would need a runtime cache invalidation
which Spec 20 does not introduce — see the deferred-items log in
``docs/08-RULES.md`` §15).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from ml.registry import list_models

router = APIRouter()


@router.get("/models")
def get_models(
    model_name: str | None = Query(
        default=None,
        description="Optional filter — only return rows for this model_name.",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum rows to return (hard-capped at 1000).",
    ),
) -> dict[str, Any]:
    """Return registry rows as JSON.

    Response shape::

        {
          "count": int,
          "models": [
            {"id": int, "model_name": str, "version": str,
             "training_date": str (ISO 8601), "is_active": bool,
             "artifact_path": str, "rmse": float | null, ...},
            ...
          ]
        }
    """
    rows = list_models(model_name=model_name, limit=limit)
    return {"count": len(rows), "models": rows}


__all__ = ["router"]
