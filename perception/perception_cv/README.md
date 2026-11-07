# perception (phone cameras)

Phones stream camera frames to this server; it finds faces, scores emotion + liking per person,
and merges all phones into one `GroupReaction` (AGENTS.md names).

## Setup
    pip install -r perception_cv/requirements.txt
    pip uninstall -y opencv-contrib-python-headless
    pip install --force-reinstall --no-deps opencv-python==5.0.0.93

deepface pulls in a second OpenCV build (`opencv-contrib-python-headless`) that shares the same `cv2`
folder as `opencv-python`; whichever installs last wins, and the headless one has no preview window.
The last two commands leave only `opencv-python`. Needs Python 3.9 or newer.

## Run
    python -m perception_cv --mock          # fake emotions, test the phone page first
    python -m perception_cv                 # real DeepFace model

### Use DroidCam as a local webcam
Start the DroidCam phone app and desktop client, connect them, and enable its virtual camera.
Then run from the `perception` directory:

    python -m perception_cv --camera 2 --detector yunet

(`2` is the DroidCam virtual-camera index on this setup; try `0` or `1` if needed.)

Other useful variants:

    python -m perception_cv --camera 2 --mock              # test camera without loading DeepFace
    python -m perception_cv --camera 2 --detector retinaface

## Use from other code
    import perception_cv
    perception_cv.serve(perception_cv.build_service(), background=True)
    reaction = perception_cv.get_group_reaction(song_id)   # GroupReaction
    # or push on the bus: service.start_publishing(lambda: current_song_id)

## Test
    python -m pytest perception_cv/tests
