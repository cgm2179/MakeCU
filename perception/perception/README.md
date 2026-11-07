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

    python -m perception --camera 2 --detector yunet

(`2` is the DroidCam virtual-camera index on this setup; try `0` or `1` if needed.)

Other useful variants:

    python -m perception --camera 2 --mock              # test camera without loading DeepFace
    python -m perception --camera 2 --detector retinaface

## Use from other code
    import perception
    perception.serve(perception.build_service(), background=True)
    reaction = perception.get_group_reaction(song_id)   # GroupReaction
    # or push on the bus: service.start_publishing(lambda: current_song_id)

## Test
    python -m pytest perception/tests
