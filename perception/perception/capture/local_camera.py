"""Read frames from a local webcam such as the DroidCam virtual camera."""
from __future__ import annotations

import sys

import cv2

from ..pipeline import PerceptionService


def _show_preview(frame) -> None:
    try:
        cv2.imshow("Perception - local camera", frame)
    except cv2.error as exc:
        if "function is not implemented" in str(exc).lower():
            raise RuntimeError(
                "OpenCV was installed without GUI support. In this venv, run "
                "`python -m pip install --force-reinstall opencv-python`, "
                "then restart perception."
            ) from exc
        raise


def run_local_camera(service: PerceptionService, camera_index: int) -> None:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(camera_index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check that the DroidCam client is running and try another --camera index."
        )

    phone_id = f"local-camera-{camera_index}"
    preview_opened = False
    try:
        print("Camera ready. Press q or Esc in the preview window to stop.")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(f"Could not read a frame from camera index {camera_index}.")

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("Could not encode the camera frame as JPEG.")
            result = service.process_frame(phone_id, encoded.tobytes())

            height, width = frame.shape[:2]
            for face in result["faces"]:
                x, y, w, h = face["box"]
                left, top = round(x * width), round(y * height)
                right, bottom = round((x + w) * width), round((y + h) * height)
                color = (0, 220, 0) if face["liking"] >= 0 else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f'{face["emotion"]} {face["liking"]:.2f}'
                cv2.putText(frame, label, (left, max(20, top - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _show_preview(frame)
            preview_opened = True
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        service.drop_phone(phone_id)
        if preview_opened:
            cv2.destroyAllWindows()
