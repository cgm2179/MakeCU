# Bill of Materials

The robot stands 1 to 1.5 ft (30–45 cm) tall and dances in place. It doesn't drive. A stepper turns the waist toward people, a second stepper turns the head, two SG90 servos wave the arms, and an 8×8 LED face shows the group's mood.

The spreadsheet version is [`BOM.xlsx`](BOM.xlsx). Sheet 1 lists the components we have, and sheet 2 lists the parts that need CAD.

## System layout

| Device | Runs | Connected to |
|---|---|---|
| Laptop | Perception (pose + emotion, two cameras) and the RL agent | Both cameras through the USB hub; sends `GroupReaction`, `UserMotion`, `NextSong` to the Pi |
| Raspberry Pi 4 (1GB) | Music system and motion system as separate processes, plus the phone link | Speaker, UNO (USB), laptop (network) |
| Elegoo UNO R3 | Motor firmware (`hardware/firmware/`): steppers, servos, face | Pi over USB serial; receives `MotionCommand` |

The UNO sits between the Pi and the motors because the Pi's software PWM makes servos jitter.

## Components

Status key: ✅ have · 🔁 backup · ↩️ return to desk · ❌ missing

| ✓ | Component | Qty | Source | Purpose | Improvise if missing |
|---|---|---|---|---|---|
| ✅ | Raspberry Pi 4 (1GB) | 1 | Desk order | Runs music playback, the message bus and the motion process; vision runs on the laptop | — |
| ✅ | KEXIN 16GB microSD | 1 | Personal | Raspberry Pi OS Lite (64-bit), flashed headless with Wi-Fi + SSH preset | — |
| ✅ | USB-C Cable | 1 | Desk order | Pi power from a wall USB-C charger | — |
| ✅ | Elegoo UNO R3 | 1 | Personal | Motor controller; takes `MotionCommand` from the Pi over USB | ESP32 (backup below) |
| ✅ | Prototype Shield v5 | 1 | Personal | Sits on the UNO; holds wiring and the 5V motor power rail | Breadboard |
| ✅ | USB A-B cable (blue) | 1 | Personal | Pi ↔ UNO serial and UNO power | — |
| ✅ | ELEGOO 28BYJ-48 Stepper Motor | 2 | Desk order | #1 turns the waist toward the group; #2 turns the head left and right | — |
| ✅ | ELEGOO ULN2003 Stepper Motor Driver | 2 | Desk order | One per stepper | — |
| ✅ | ELEGOO SG90 Micro Servo | 2 | 1 desk order + 1 personal | Left and right shoulders: wave the arms to the beat. Keep each arm under ~15 g | — |
| ✅ | MAX7219 8×8 dot matrix module | 1 | Personal | Face that shows `group_emotion` | — |
| ✅ | Sound sensor module (mic + pot) | 1 | Personal | Optional: beat detection so the arms dance in time | — |
| ✅ | Bluetooth speaker | 1 | Personal | Plays music from the Pi; use AUX into the Pi's 3.5mm jack if it has one | — |
| ✅ | Power bank | 1 | Personal | 5V for all steppers and servos (2A+ port), via a stripped USB cable to the shield rail | — |
| ✅ | Jumper wires | 1 bundle | Personal | UNO ↔ drivers, servos, face | — |
| ✅ | Component box (capacitors, resistors, LEDs) | 1 | Personal | 1000µF capacitor across the servo rail; status LEDs | — |
| ✅ | Benfei USB hub | 1 | Personal | Both cameras into the laptop | — |
| ✅ | Laptop | 1 | Personal | Perception + RL | — |
| 🔁 | ESP32 ESP-WROOM-32 Development Board | 1 | Desk order | Backup motor controller (Wi-Fi capable, 3.3V logic) | — |
| 🔁 | Elegoo 9V adapter + breadboard power module | 1 | Personal | Bench testing only (~0.7A limit) | — |
| ↩️ | TT Gearbox motor | 1 | Desk order | Not needed: the robot doesn't drive | — |
| ↩️ | L298N motor driver | 1 | Desk order | Not needed without the TT motor | — |
| ↩️ | Wheels | 1 | Desk order | Not needed: the robot doesn't drive | — |
| ❌ | Camera #1 | 0 | — | Emotion + pose (`CAMERA_LEFT_INDEX`), mounted in the head | USB Cameras from the desk, or a phone as a webcam |
| ❌ | Camera #2 | 0 | — | Second view for more people (`CAMERA_RIGHT_INDEX`) | Logitech Webcams (MLH desk), or a phone (Continuity Camera / DroidCam / Iriun) |
| ❌ | Sunfounder 55 g Servo | 0 | On desk order 0-00038, not received | Would have given stronger shoulders | Ask the desk; otherwise the SG90s cover the shoulders |
| ❌ | Extra micro servos (optional) | 0 | — | Head tilt and an elbow | **9G Micro Servo** (MLH desk); without them, fix the cameras ~10–15° up and skip elbows |
| ❌ | USB-C wall charger (5V, 2.5A+) | 0 | — | Powers the Pi | Any phone charger; or a 2-port power bank (3A+ total) shared with the motors |
| ❌ | Waist bearing | 0 | — | Carries the torso so the stepper only rotates it | 3D-printed ball-bearing ring, or a lazy susan / two smooth plates with washers |
| ❌ | Chassis | 0 | — | Holds everything at 1–1.5 ft | See the CAD sheet; cardboard + hot glue as a fallback |

Not used: HC-SR04 ultrasonic, HC-SR501 PIR, MPU-9250 IMU, DS1307 RTC, rotary encoder, small DC motor, Grove kit. RFID-RC522 is an optional extra ("vibe cards" that set the event filter) and needs the music/app owner's sign-off.

## Power

- **Pi:** wall USB-C charger. The robot is cabled to the laptop for the cameras anyway.
- **Motors and servos:** power bank → stripped USB cable → shield 5V rail, with a 1000µF capacitor across it.
- **UNO:** powered from the Pi over USB.
- **Grounds:** tie the power bank ground to the UNO ground. Never power servos from the UNO's 5V pin.

## UNO pin plan

| Part | Pins |
|---|---|
| Waist stepper (ULN2003 #1) | D2, D3, D4, D5 |
| Head-pan stepper (ULN2003 #2) | D6, D7, D8, D12 |
| Left / right shoulder SG90 | D9, D10 |
| Spare: head-tilt / elbow servo if you get more | D11, A0 |
| MAX7219 face (DIN, CS, CLK) | A1, A2, A3 |
| Sound sensor (AO) | A4 |

Libraries: `AccelStepper`, `Servo`, `LedControl`.

## Parts to CAD

The full list with dimensions is in sheet 2 of [`BOM.xlsx`](BOM.xlsx). In short: a base that holds the speaker, power bank, Pi, UNO and both drivers; a waist turntable; the torso with two SG90 shoulder pockets; two light arms; a neck stepper mount; and a head with two camera mounts angled outward and fixed ~10–15° up, plus a face window.
