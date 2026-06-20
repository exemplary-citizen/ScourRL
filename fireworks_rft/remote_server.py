from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cart_scout.reward import score_purchase_packet
from cart_scout.schema import ShoppingTaskSpec

app = FastAPI(title="CartScout Remote RFT")


class RolloutRequest(BaseModel):
    task: ShoppingTaskSpec
    final_answer: str


@app.post("/init")
async def init(request: RolloutRequest):
    """Minimal remote-browser RFT shape.

    The hackathon MVP can post a completed rollout here. The stretch version should
    start a HUD/browser rollout inside this handler and grade its final packet.
    """
    try:
        result = score_purchase_packet(request.final_answer, request.task)
        return {
            "status": "success",
            "reward": result.score,
            "breakdown": result.breakdown,
            "reasons": result.reasons,
        }
    except Exception as exc:  # pragma: no cover - defensive API boundary
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
