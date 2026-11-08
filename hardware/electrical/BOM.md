# Bill of Materials

The robot stands 1 to 1.5 ft (30–45 cm) tall and dances in place. A stepper turns the waist toward people, a second stepper turns the head, two SG90 servos wave the arms, a micro servo tilts the head, and an 8×8 LED face shows the group's mood. Driving around is a stretch goal.

The spreadsheet version is [`BOM.xlsx`](BOM.xlsx):
- **All Components**: everything on hand (desk order, personal inventory, a teammate's OSOYOO Model-3 kit) with a status for each.
- **Plan to Use**: what goes into the robot, its pins and power, plus what is still missing.
- **Components to CAD**: printed and cut parts with key dimensions.

## System layout

| Device | Runs | Connected to |
|---|---|---|
| Laptop | Perception (pose + emotion, two cameras) and the RL agent | Both cameras through the USB hub; sends `GroupReaction`, `UserMotion`, `NextSong` to the Pi |
| Raspberry Pi 4 (1GB) | Music system and motion system as separate processes, plus the phone link | Speaker, UNO (USB), laptop (network) |
| Elegoo UNO R3 | Motor firmware (`hardware/firmware/`): steppers, servos, face | Pi over USB serial; receives `MotionCommand` |

The UNO sits between the Pi and the motors because the Pi's software PWM makes servos jitter.

## All components

Status key: ✅ use · 🟡 optional · 🔁 backup · ↩️ return to desk · ⛔ not used

| ✓ | Component | Qty | Source |
|---|---|---|---|
| ✅ | Raspberry Pi 4 (1GB) | 1 | Desk order |
| ✅ | USB-C Cable | 1 | Desk order |
| ✅ | ELEGOO 28BYJ-48 Stepper Motor | 2 | Desk order |
| ✅ | ELEGOO ULN2003 Stepper Motor Driver | 2 | Desk order |
| ✅ | ELEGOO SG90 Micro Servo | 1 | Desk order |
| 🔁 | ESP32 ESP-WROOM-32 Development Board | 1 | Desk order |
| ↩️ | TT Gearbox motor | 1 | Desk order |
| ↩️ | L298N motor driver | 1 | Desk order |
| ↩️ | Wheels | 1 | Desk order |
| ✅ | Laptop | 1 | Personal |
| ✅ | Benfei USB hub | 1 | Personal |
| ✅ | Bluetooth speaker | 1 | Personal |
| ✅ | Power bank | 1 | Personal |
| ✅ | KEXIN 16GB microSD | 1 | Personal |
| ✅ | Elegoo UNO R3 | 1 | Personal |
| ✅ | Prototype Shield v5 (with mini breadboard) | 1 | Personal |
| ✅ | USB A-B cable (blue) | 1 | Personal |
| ✅ | ELEGOO SG90 Micro Servo | 1 | Personal |
| ✅ | MAX7219 8×8 dot matrix module | 1 | Personal |
| 🟡 | Sound sensor module (mic + pot) | 1 | Personal |
| ✅ | Jumper wires | 1 bundle | Personal |
| ✅ | Component box (capacitors, resistors, LEDs) | 1 | Personal |
| 🔁 | Elegoo 9V 1A AC/DC adapter | 1 | Personal |
| 🔁 | Elegoo breadboard power supply module | 1 | Personal |
| 🟡 | RFID-RC522 + card + key fob | 1 | Personal |
| ⛔ | Grove cables, HC-SR04 ultrasonic, HC-SR501 PIR, MPU-9250 IMU, DS1307 RTC, rotary encoder, small DC motor, MLH Grove kit | 1 each | Personal |
| ✅ | Micro servo (blue, SG90-class) | 1 | OSOYOO kit (teammate) |
| ✅ | Acrylic servo pan/tilt bracket | 1 | OSOYOO kit (teammate) |
| ✅ | Servo horns + screws | 1 | OSOYOO kit (teammate) |
| 🟡 | Dual-motor gearbox (2 DC motors) | 1 | OSOYOO kit (teammate) |
| 🟡 | Wheels (red, rubber tires) | 2 | OSOYOO kit (teammate) |
| 🟡 | OSOYOO motor driver shield | 1 | OSOYOO kit (teammate) |
| 🟡 | Green rechargeable battery pack (check voltage) | 1 | OSOYOO kit (teammate) |
| 🔁 | 9V battery (Amazon Basics alkaline) | 1 | OSOYOO kit (teammate) |
| ⛔ | Small board with built-in screen (looks like an ESP32 Feather with TFT) | 1 | OSOYOO kit (teammate) |
| ⛔ | Rest of the OSOYOO Model-3 kit (not inventoried yet) | — | OSOYOO kit (teammate) |

## Plan to use

| Component | Role | Pins / connection | Improvise if missing |
|---|---|---|---|
| Laptop + Benfei USB hub | Perception + RL | Cameras (USB), Pi (network) | — |
| Raspberry Pi 4 + KEXIN 16GB microSD | Music + motion processes, phone link | Speaker, UNO (USB), laptop | — |
| Elegoo UNO R3 + Prototype Shield v5 | Motor firmware | Pi via the blue USB A-B cable | ESP32 |
| 28BYJ-48 + ULN2003 #1 | Waist turn toward the group | D2, D3, D4, D5 | — |
| 28BYJ-48 + ULN2003 #2 | Head turn left and right | D6, D7, D8, D12 | — |
| 2× ELEGOO SG90 | Left and right shoulders | D9, D10 | — |
| OSOYOO micro servo + acrylic bracket | Head tilt (look up at faces) | D11 | Fix the cameras ~10–15° up |
| MAX7219 8×8 matrix | Face showing `group_emotion` | A1 (DIN), A2 (CS), A3 (CLK) | — |
| Sound sensor (optional) | Beat detection | A4 | — |
| Bluetooth speaker | Music | Pi (AUX into the 3.5mm jack if it has one) | — |
| Power bank | 5V for steppers + servos (2A+ port) | Stripped USB cable to the shield rail | Green battery pack if it's 4.8–6V |

Spare pins: A0 (elbow servo if one turns up), A5, D13.

### Still missing

| Component | Purpose | Improvise |
|---|---|---|
| Camera #1 | Emotion + pose (`CAMERA_LEFT_INDEX`) | USB Cameras from the desk, or a phone as a webcam |
| Camera #2 | Second view for more people (`CAMERA_RIGHT_INDEX`) | Logitech Webcams (MLH desk), or a phone (Continuity Camera / DroidCam / Iriun) |
| USB-C wall charger (5V, 2.5A+) | Powers the Pi | Any phone charger |
| Waist bearing | Carries the torso so the stepper only rotates it | 3D-printed ball race, or a lazy susan / two smooth plates with washers |
| Chassis | Holds everything at 1–1.5 ft | See the CAD sheet; cardboard + hot glue as a fallback |
| Sunfounder 55 g Servo (on order 0-00038, not received) | Stronger shoulders | Ask the desk; the SG90s cover the shoulders |
| Extra micro servo (optional) | Elbow | 9G Micro Servo (MLH desk); otherwise skip elbows |

## Power

- **Pi:** wall USB-C charger. The robot is cabled to the laptop for the cameras anyway.
- **Motors and servos:** power bank → stripped USB cable → shield 5V rail, with a 1000µF capacitor across it.
- **UNO:** powered from the Pi over USB.
- **Grounds:** tie the power bank ground to the UNO ground. Never power servos from the UNO's 5V pin.

Libraries: `AccelStepper`, `Servo`, `LedControl`.

## Parts to CAD

The full list with dimensions is in the **Components to CAD** sheet of [`BOM.xlsx`](BOM.xlsx). In short:
- a base for the speaker, power bank, Pi, UNO and both drivers
- a waist turntable with a stepper mount and shaft couplers
- the torso with two SG90 shoulder pockets, and two light arms
- a neck stepper mount
- a head with two angled camera mounts, the tilt hinge (the OSOYOO acrylic bracket works until it's printed) and a face window
- stretch goal: a drive-base mount for the OSOYOO gearbox and wheels
