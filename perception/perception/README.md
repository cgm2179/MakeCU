# perception (phone cameras)

Phones stream camera frames to this server; it finds faces, scores emotion + liking per person,
and merges all phones into one `GroupReaction` (AGENTS.md names).

## Run
    pip install -r perception/requirements.txt
    python -m perception --mock          # fake emotions, test the phone page first
    python -m perception                 # real DeepFace model

### Use DroidCam as a local webcam
Start the DroidCam phone app and desktop client, connect them, and enable its virtual camera.
Then run from the `perception` directory:

    python -m perception --camera 0 --mock   # try the camera without loading DeepFace
    python -m perception --camera 0           # use the real DeepFace model

Replace `0` with the DroidCam camera device index if needed. The local preview shows face
boxes and scores; press `q` or Esc in that window to exit. This mode uses the PC's
DroidCam virtual-camera device and does not require the phone webpage or HTTPS setup.
If you see an OpenCV error saying `The function is not implemented` at `cv2.imshow`,
reinstall the GUI-enabled OpenCV package in the active virtual environment:

    python -m pip install --force-reinstall opencv-python

Phones need HTTPS for camera access. Pick one:
- Tunnel (easiest, also dodges venue Wi-Fi client isolation):  cloudflared tunnel --url http://localhost:8000
  then open the printed https URL on each phone.
- Self-signed: python -m perception.capture.make_cert, rerun `python -m perception`,
  open https://<laptop-ip>:8000 on the phone and accept the warning.

## Use from other code
    import perception
    perception.serve(perception.build_service(), background=True)
    reaction = perception.get_group_reaction(song_id)   # GroupReaction
    # or push on the bus: service.start_publishing(lambda: current_song_id)

## Test
    python -m pytest perception/tests
