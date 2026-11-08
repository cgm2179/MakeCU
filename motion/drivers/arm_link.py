"""Send the current arm profile to the Arduino over serial.

Line format (one update per line):

    ARM <period_s> <amplitude> <phase_offset_deg> <emotion>

Example:

    ARM 0.550 0.850 180.0 happy

If port is None, the line is stored in last_line and nothing is sent.
That is what tests and a laptop with no Arduino plugged in use.
"""
from __future__ import annotations

from motion.control.arm_params import ArmProfile


class ArmLink:
    def __init__(self, port: str | None = None, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.last_line = ""
        self._ser = None
        if port:
            import serial  # pyserial; only needed when an Arduino is attached
            self._ser = serial.Serial(port, baud, timeout=1)

    def send(self, profile: ArmProfile, emotion: str) -> str:
        line = (f"ARM {profile.period_s:.3f} {profile.amplitude:.3f} "
                f"{profile.phase_offset_deg:.1f} {emotion}")
        self.last_line = line
        if self._ser is not None:
            self._ser.write((line + "\n").encode())
        return line

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
