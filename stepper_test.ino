/*
  Simple test: two 28BYJ-48 steppers (ULN2003) on ESP32. No libraries needed.

  Wiring (ESP32 GPIO -> ULN2003 IN1..IN4, in order):
  Power: ULN2003 "+" and "-" from a 5V supply (power bank), NOT the ESP32 3.3V.
         Connect the supply GND to the ESP32 GND.

  What you should see (repeats forever):
    1. Left stepper turns 1/4 rev one way, then back
    2. Right stepper turns 1/4 rev one way, then back
    3. Both turn together, then back
  Serial Monitor (115200) prints what it is doing.
*/

const int LEFT_PINS[4]  = {13, 12, 14, 27};
const int RIGHT_PINS[4] = {26, 25, 33, 32};

// Half-step sequence (8 phases) for 28BYJ-48
const int SEQ[8][4] = {
  {1, 0, 0, 0},
  {1, 1, 0, 0},
  {0, 1, 0, 0},
  {0, 1, 1, 0},
  {0, 0, 1, 0},
  {0, 0, 1, 1},
  {0, 0, 0, 1},
  {1, 0, 0, 1}
};

const int STEPS_QUARTER_TURN = 1024;  // 4096 half-steps = 1 full revolution
const int STEP_DELAY_MS = 2;          // increase (e.g. 3-5) if the motor buzzes or skips

int leftPhase = 0;
int rightPhase = 0;

void applyPhase(const int pins[4], int phase) {
  for (int i = 0; i < 4; i++) digitalWrite(pins[i], SEQ[phase][i]);
}

void releaseCoils(const int pins[4]) {
  for (int i = 0; i < 4; i++) digitalWrite(pins[i], LOW);
}

// dir = +1 or -1
void stepLeft(int dir)  { leftPhase  = (leftPhase  + dir + 8) % 8; applyPhase(LEFT_PINS,  leftPhase); }
void stepRight(int dir) { rightPhase = (rightPhase + dir + 8) % 8; applyPhase(RIGHT_PINS, rightPhase); }

void moveLeft(int steps, int dir) {
  for (int i = 0; i < steps; i++) { stepLeft(dir); delay(STEP_DELAY_MS); }
  releaseCoils(LEFT_PINS);
}

void moveRight(int steps, int dir) {
  for (int i = 0; i < steps; i++) { stepRight(dir); delay(STEP_DELAY_MS); }
  releaseCoils(RIGHT_PINS);
}

void moveBoth(int steps, int dir) {
  for (int i = 0; i < steps; i++) {
    stepLeft(dir);
    stepRight(dir);
    delay(STEP_DELAY_MS);
  }
  releaseCoils(LEFT_PINS);
  releaseCoils(RIGHT_PINS);
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) {
    pinMode(LEFT_PINS[i], OUTPUT);
    pinMode(RIGHT_PINS[i], OUTPUT);
  }
  releaseCoils(LEFT_PINS);
  releaseCoils(RIGHT_PINS);
  Serial.println("Stepper test starting in 2 seconds...");
  delay(2000);
}

void loop() {
  Serial.println("LEFT forward");
  moveLeft(STEPS_QUARTER_TURN, +1);
  delay(500);
  Serial.println("LEFT back");
  moveLeft(STEPS_QUARTER_TURN, -1);
  delay(1000);

  Serial.println("RIGHT forward");
  moveRight(STEPS_QUARTER_TURN, +1);
  delay(500);
  Serial.println("RIGHT back");
  moveRight(STEPS_QUARTER_TURN, -1);
  delay(1000);

  Serial.println("BOTH forward");
  moveBoth(STEPS_QUARTER_TURN, +1);
  delay(500);
  Serial.println("BOTH back");
  moveBoth(STEPS_QUARTER_TURN, -1);
  delay(2000);
}
