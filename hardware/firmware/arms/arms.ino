// Arm driver. Listens for lines from motion/drivers/arm_link.py:
//
//   ARM <period_s> <amplitude> <phase_offset_deg> <emotion>
//
// Both servos oscillate until a new line arrives.
// phase 0   = arms together
// phase 180 = arms opposite
//
// Change the pins and the pulse limits to match the build.

#include <Servo.h>

const int LEFT_PIN = 9;
const int RIGHT_PIN = 10;
const int CENTER_US = 1500;
const int MAX_SWING_US = 400;  // amplitude 1.0 swings +/- this many microseconds

Servo leftServo;
Servo rightServo;

float periodS = 1.40;
float amplitude = 0.40;
float phaseDeg = 0.0;
String emotion = "neutral";

void setup() {
  Serial.begin(115200);
  leftServo.attach(LEFT_PIN);
  rightServo.attach(RIGHT_PIN);
  leftServo.writeMicroseconds(CENTER_US);
  rightServo.writeMicroseconds(CENTER_US);
}

void applyLine(String line) {
  if (!line.startsWith("ARM ")) return;
  // ARM period amplitude phase emotion
  float p, a, ph;
  char emo[16] = "neutral";
  if (sscanf(line.c_str(), "ARM %f %f %f %15s", &p, &a, &ph, emo) == 4) {
    periodS = p > 0.05f ? p : 0.05f;
    amplitude = constrain(a, 0.0f, 1.0f);
    phaseDeg = ph;
    emotion = emo;
  }
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    applyLine(line);
  }

  float t = millis() / 1000.0f;
  float w = TWO_PI / periodS;
  float phase = phaseDeg * DEG_TO_RAD;
  int swing = (int)(amplitude * MAX_SWING_US);
  int leftUs = CENTER_US + (int)(swing * sin(w * t));
  int rightUs = CENTER_US + (int)(swing * sin(w * t + phase));
  leftServo.writeMicroseconds(leftUs);
  rightServo.writeMicroseconds(rightUs);
  delay(20);
}
