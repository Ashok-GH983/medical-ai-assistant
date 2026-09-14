# =============================================================================
# main.py — FastAPI backend (REST + SSE streaming) for the medical literature AI
# =============================================================================

import os
import uuid
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal, engine, Base, ResearchSession
from agent_pipeline import app_pipeline

# -----------------------------------------------------------------------------
# Logging + DB bootstrap
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-ai-backend")

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Medical Literature AI Assistant API",
    description="Backend API powering an autonomous LangGraph multi-agent "
                "literature review & RAG synthesis pipeline.",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# CORS — env-controlled allow-list (defaults to * for dev)
# -----------------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    query: str
    run_id: Optional[str] = None


class ReviewResponse(BaseModel):
    run_id: str
    query: str
    status: str
    review_status: str
    synthesis: str
    keywords: List[str]
    sub_questions: List[str]
    pmids: List[str]
    papers: List[Dict[str, Any]]
    extractions: List[Dict[str, Any]]
    dataset_comparison: List[Dict[str, Any]]
    method_comparison: List[Dict[str, Any]]
    metric_summary: List[Dict[str, Any]]
    research_gaps: List[str]
    references: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# DB session dependency (used by the non-streaming endpoint)
# -----------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {
        "message": "Medical Literature AI Assistant API is up and running.",
        "status": "healthy",
    }


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def _initial_state(run_id: str, query: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "query": query,
        "keywords": [],
        "sub_questions": [],
        "pmids": [],
        "raw_papers": [],
        "screened_papers": [],
        "extractions": [],
        "dataset_comparison": [],
        "method_comparison": [],
        "metric_summary": [],
        "research_gaps": [],
        "synthesis": "",
        "references": [],
        "review_status": "",
        "status": "Starting",
    }


def _build_response_payload(run_id: str, query: str, final_state: Dict[str, Any]) -> Dict[str, Any]:
    screened_papers = final_state.get("screened_papers", [])
    references = [
        {
            "pmid": p.get("pmid"),
            "title": p.get("title"),
            "authors": p.get("authors", "Unknown Authors"),
            "journal": p.get("journal", ""),
            "pubdate": p.get("pubdate", "N/A"),
        }
        for p in screened_papers
    ]
    return {
        "run_id": run_id,
        "query": query,
        "status": "Completed",
        "review_status": final_state.get("review_status", "Not reviewed"),
        "synthesis": final_state.get("synthesis", "No synthesis generated."),
        "keywords": final_state.get("keywords", []),
        "sub_questions": final_state.get("sub_questions", []),
        "pmids": final_state.get("pmids", []),
        "papers": screened_papers,
        "extractions": final_state.get("extractions", []),
        "dataset_comparison": final_state.get("dataset_comparison", []),
        "method_comparison": final_state.get("method_comparison", []),
        "metric_summary": final_state.get("metric_summary", []),
        "research_gaps": final_state.get("research_gaps", []),
        "references": references,
    }


def _persist_session(run_id: str, query: str, final_state: Dict[str, Any]) -> None:
    """Save a ResearchSession row using a fresh DB session (stream-safe)."""
    try:
        db = SessionLocal()
        db.add(
            ResearchSession(
                run_id=run_id,
                query=query,
                keywords=",".join(final_state.get("keywords", [])) or "",
                pmid_count=len(final_state.get("pmids", [])),
                synthesis=final_state.get("synthesis", ""),
            )
        )
        db.commit()
        db.close()
        logger.info(f"[DB] Saved session {run_id}")
    except Exception as e:
        logger.error(f"[DB] Save failed for {run_id}: {e}")


# -----------------------------------------------------------------------------
# Endpoint 1 — synchronous (original behaviour preserved)
# -----------------------------------------------------------------------------
@app.post("/api/review", response_model=ReviewResponse)
async def execute_literature_review(request: ReviewRequest, db: Session = Depends(get_db)):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    run_id = request.run_id or str(uuid.uuid4())
    logger.info(f"Run {run_id} | Query: '{request.query}'")

    initial_state = _initial_state(run_id, request.query)

    try:
        final_state = app_pipeline.invoke(initial_state)
        payload = _build_response_payload(run_id, request.query, final_state)

        db.add(
            ResearchSession(
                run_id=run_id,
                query=request.query,
                keywords=",".join(payload["keywords"]),
                pmid_count=len(payload["pmids"]),
                synthesis=payload["synthesis"],
            )
        )
        db.commit()

        return ReviewResponse(**payload)

    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


# -----------------------------------------------------------------------------
# Endpoint 2 — SSE streaming (frontend uses this)
# -----------------------------------------------------------------------------
NODE_STAGE_MAP = {
    "planner":     "Searching",
    "retriever":   "Searching",
    "screener":    "Screening",
    "extractor":   "Extracting",
    "comparator":  "Comparing",
    "synthesizer": "Synthesizing",
    "reviewer":    "Completed",
}


@app.post("/api/review/stream")
async def execute_literature_review_stream(request: ReviewRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    run_id = request.run_id or str(uuid.uuid4())
    logger.info(f"[STREAM] Run {run_id} | Query: '{request.query}'")

    initial_state = _initial_state(run_id, request.query)

    async def event_generator():
        final_state = dict(initial_state)
        try:
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'Searching', 'node': 'planner'})}\n\n"

            loop = asyncio.get_event_loop()
            queue: "asyncio.Queue" = asyncio.Queue()

            def run_pipeline():
                try:
                    for step in app_pipeline.stream(initial_state):
                        loop.call_soon_threadsafe(queue.put_nowait, ("step", step))
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

            loop.run_in_executor(None, run_pipeline)

            while True:
                kind, payload = await queue.get()

                if kind == "step":
                    for node_name, node_output in payload.items():
                        if isinstance(node_output, dict):
                            final_state.update(node_output)
                        stage = NODE_STAGE_MAP.get(node_name, node_name.capitalize())
                        yield f"data: {json.dumps({'type': 'stage', 'stage': stage, 'node': node_name})}\n\n"

                elif kind == "done":
                    break

                elif kind == "error":
                    logger.error(f"[STREAM] Error on {run_id}: {payload}")
                    yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                    return

            _persist_session(run_id, request.query, final_state)
            result_payload = _build_response_payload(run_id, request.query, final_state)
            yield f"data: {json.dumps({'type': 'result', 'data': result_payload})}\n\n"

        except Exception as exc:
            logger.error(f"[STREAM] Fatal error on {run_id}: {exc}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)