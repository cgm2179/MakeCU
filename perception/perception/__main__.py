"""python -m perception [--mock] [--camera INDEX] [--cert certs/cert.pem --key certs/key.pem]"""
import argparse
import os

from .api import build_service, serve
from .capture.netutil import local_ip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cert")
    ap.add_argument("--key")
    ap.add_argument("--mock", action="store_true", help="fake emotions, no ML needed")
    ap.add_argument("--detector", default="opencv", help="opencv | ssd | mediapipe | yunet | retinaface")
    ap.add_argument("--camera", type=int, help="read from a local camera device index (e.g. DroidCam)")
    args = ap.parse_args()

    service = build_service(mock=args.mock, detector=args.detector)
    if args.camera is not None:
        from .capture.local_camera import run_local_camera
        run_local_camera(service, args.camera)
        return

    if not args.cert and os.path.exists("certs/cert.pem"):
        args.cert, args.key = "certs/cert.pem", "certs/key.pem"
    scheme = "https" if args.cert else "http"
    print(f"Open on each phone: {scheme}://{local_ip()}:{args.port}/   (debug JSON: /reaction)")
    if not args.cert:
        print("No cert found: phones will block the camera unless you use localhost or an https tunnel.")
    serve(service, args.host, args.port, args.cert, args.key)


if __name__ == "__main__":
    main()
