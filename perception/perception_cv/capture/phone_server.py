"""Serves the phone page and receives camera frames over a WebSocket.

Flow control: the phone sends one JPEG, waits for our JSON reply (faces + scores), then sends the next.
So a slow model never builds a backlog and phones adapt their frame rate automatically.
"""
from __future__ import annotations
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..pipeline import PerceptionService

STATIC = Path(__file__).parent / "static"


def create_app(service: PerceptionService) -> FastAPI:
    app = FastAPI(title="perception phone capture")
    pool = ThreadPoolExecutor(max_workers=1)   # one inference at a time, shared by all phones

    @app.get("/")
    def index():
        return FileResponse(STATIC / "phone.html")

    @app.get("/health")
    def health():
        return {"phones": service.phone_count()}

    @app.get("/reaction")
    def reaction(song_id: str = "debug"):
        return service.get_group_reaction(song_id).to_dict()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket, phone: Optional[str] = None):
        await websocket.accept()
        phone_id = phone or uuid.uuid4().hex[:6]
        loop = asyncio.get_running_loop()
        try:
            while True:
                data = await websocket.receive_bytes()
                reply = await loop.run_in_executor(pool, service.process_frame, phone_id, data)
                await websocket.send_json(reply)
        except WebSocketDisconnect:
            pass
        except Exception as e:  # keep the server alive on a bad client
            print(f"[perception] phone {phone_id} error: {e!r}")
        finally:
            service.drop_phone(phone_id)

    return app
