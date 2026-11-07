"""Public entry points for the rest of the repo (see AGENTS.md: get_group_reaction)."""
from __future__ import annotations
import threading

from .config import PerceptionConfig
from .pipeline import PerceptionService
from .types import GroupReaction

_service: PerceptionService | None = None


def build_service(mock: bool = False, detector: str = "opencv", cfg: PerceptionConfig | None = None) -> PerceptionService:
    if mock:
        from .emotion.mock_model import MockModel
        model = MockModel()
    else:
        from .emotion.deepface_model import DeepFaceModel
        model = DeepFaceModel(detector_backend=detector)
    return PerceptionService(model, cfg)


def serve(service: PerceptionService, host: str = "0.0.0.0", port: int = 8000,
          certfile: str | None = None, keyfile: str | None = None, background: bool = False):
    """Run the phone server. With background=True it runs in a daemon thread and returns."""
    global _service
    import uvicorn
    from .capture.phone_server import create_app

    _service = service
    config = uvicorn.Config(create_app(service), host=host, port=port,
                            ssl_certfile=certfile, ssl_keyfile=keyfile, log_level="warning")
    server = uvicorn.Server(config)
    if background:
        threading.Thread(target=server.run, daemon=True).start()
        return server
    server.run()


def get_group_reaction(song_id: str) -> GroupReaction:
    if _service is None:
        raise RuntimeError("perception_cv not started: call perception_cv.serve(...) first")
    return _service.get_group_reaction(song_id)
