# perception (phone cameras)

Phones stream camera frames to this server; it finds faces, scores emotion + liking per person,
and merges all phones into one `GroupReaction` (AGENTS.md names).

## Setup
    pip install -r perception_cv/requirements.txt
    pip uninstall -y opencv-contrib-python-headless
    pip install --force-reinstall --no-deps "opencv-python<5"

deepface pulls in a second OpenCV build (`opencv-contrib-python-headless`) that shares the same `cv2`
folder as `opencv-python`; whichever installs last wins, and the headless one has no preview window.
The last two commands leave only `opencv-python` (4.x: OpenCV 5 dropped `CascadeClassifier`, which DeepFace's default `opencv` detector needs). Needs Python 3.9 or newer.

## Run
    python -m perception_cv --mock          # fake emotions, test the phone page first
    python -m perception_cv                 # real DeepFace model

### Use DroidCam as a local webcam
Start the DroidCam phone app and desktop client, connect them, and enable its virtual camera.
Then run from the `perception` directory:

    python -m perception_cv --camera 2 --detector yunet

(`2` is the DroidCam virtual-camera index on this setup; try `0` or `1` if needed.)

### Tune or disable detected emotions
Edit `perception_cv/emotion/tuning.py`:

- Add or remove raw emotion names in `ENABLED_RAW_EMOTIONS`. Only happy, sad, and
  neutral are enabled by default; the other DeepFace classes are commented out.
- Set `MIN_PROBABILITY` values from `0.0` to `1.0` to ignore weak scores for each
  enabled emotion. For example, `"happy": 0.6` requires a happy score of at least
  60 percent. Kept scores are renormalized; if none pass, that face is omitted.
- Restart the program after editing. This is a probability filter, not model
  retraining, so it cannot make DeepFace intrinsically more accurate.

“Bored” is derived from a still face rather than detected by DeepFace. Disable it
with `ENABLE_BORED = False` in `perception_cv/config.py`.

Arm motion profiles are tuned separately in `motion/control/arm_params.py`.
Change `HAPPY`, `SAD`, and `NEUTRAL` there to adjust period, amplitude, and phase.

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
