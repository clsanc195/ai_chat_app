import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from graph import SUMMARY_THRESHOLD, graph

app = FastAPI(title="AI Chat App")


# ---------- Request models ----------

class CreateThreadRequest(BaseModel):
    metadata: dict | None = None


class CreateRunRequest(BaseModel):
    content: str
    assistant_id: str = "asst_default"
    model: str | None = None
    instructions: str | None = None
    metadata: dict | None = None


# ---------- Helpers ----------

def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _now() -> int:
    return int(time.time())


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _role(message_type: str) -> str:
    return {"human": "user", "ai": "assistant"}.get(message_type, message_type)


def _to_message_dict(m, thread_id: str) -> dict:
    return {
        "id": getattr(m, "id", None) or _id("msg"),
        "object": "thread.message",
        "created_at": _now(),
        "thread_id": thread_id,
        "role": _role(m.type),
        "content": [{"type": "text", "text": {"value": m.content, "annotations": []}}],
    }


# ---------- Threads ----------

@app.post("/v1/threads")
def create_thread(body: CreateThreadRequest | None = None):
    return {
        "id": _id("thread"),
        "object": "thread",
        "created_at": _now(),
        "metadata": (body.metadata if body else None) or {},
    }


@app.get("/v1/threads/{thread_id}")
def get_thread(thread_id: str):
    state = graph.get_state(_thread_config(thread_id))
    if not state.values:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {
        "id": thread_id,
        "object": "thread",
        "created_at": _now(),
        "metadata": {},
    }


@app.delete("/v1/threads/{thread_id}")
def delete_thread(thread_id: str):
    # TODO: MemorySaver has no public delete API; revisit when swapping to SqliteSaver
    return {"id": thread_id, "object": "thread.deleted", "deleted": True}


# ---------- Messages ----------

@app.get("/v1/threads/{thread_id}/messages")
def list_messages(thread_id: str, limit: int = 20, order: Literal["asc", "desc"] = "desc"):
    state = graph.get_state(_thread_config(thread_id))
    messages = state.values.get("messages", []) if state.values else []

    data = [_to_message_dict(m, thread_id) for m in messages]
    if order == "desc":
        data = list(reversed(data))
    data = data[:limit]

    return {
        "object": "list",
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": False,
    }


# ---------- Runs ----------

@app.post("/v1/threads/{thread_id}/runs")
def create_run(thread_id: str, body: CreateRunRequest):
    config = _thread_config(thread_id)
    result = graph.invoke(
        {"messages": [HumanMessage(content=body.content)]},
        config=config,
    )
    last = result["messages"][-1]
    # To view the message cound and the summary
    state = graph.get_state(config)
    print(f"[debug] msg count: {len(state.values.get('messages', []))}, "
          f"summary: {state.values.get('summary', '(none)')}")
    return {
        "id": _id("run"),
        "object": "thread.run",
        "created_at": _now(),
        "thread_id": thread_id,
        "assistant_id": body.assistant_id,
        "status": "completed",
        "model": body.model or "claude-opus-4-7",
        "instructions": body.instructions,
        "metadata": body.metadata or {},
        "message": _to_message_dict(last, thread_id),
    }


@app.get("/v1/threads/{thread_id}/runs/{run_id}")
def get_run(thread_id: str, run_id: str):
    # TODO: runs aren't persisted as first-class objects yet; need a runs table/store
    return {
        "id": run_id,
        "object": "thread.run",
        "created_at": _now(),
        "thread_id": thread_id,
        "assistant_id": "asst_default",
        "status": "completed",
        "model": "claude-opus-4-7",
    }


# ---------- Graph visualization ----------

def _theme_mermaid_dark(src: str) -> str:
    return (
        src
        .replace("fill:#f2f0ff", "fill:#312e81,stroke:#818cf8,color:#e0e7ff")
        .replace("fill:#bfb6fc", "fill:#4f46e5,stroke:#c7d2fe,color:#ffffff")
        .replace(
            "classDef first fill-opacity:0",
            "classDef first fill-opacity:0,stroke:#a5b4fc,color:#e0e7ff",
        )
    )


_GRAPH_HTML = (
    (Path(__file__).parent / "resources" / "graph.html")
    .read_text(encoding="utf-8")
    .replace("__MERMAID__", _theme_mermaid_dark(graph.get_graph().draw_mermaid()))
    .replace("__SUMMARY_THRESHOLD__", str(SUMMARY_THRESHOLD))
)

_CHAT_HTML = (Path(__file__).parent / "resources" / "chat.html").read_text(encoding="utf-8")


@app.get("/graph", response_class=HTMLResponse)
def view_graph():
    return HTMLResponse(content=_GRAPH_HTML)


@app.get("/chat", response_class=HTMLResponse)
def view_chat():
    return HTMLResponse(content=_CHAT_HTML)
