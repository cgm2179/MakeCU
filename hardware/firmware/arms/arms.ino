// Arm driver for two 28BYJ-48 steppers (ULN2003).
// Listens for the same lines the Python side already sends:
//
//   ARM <period_s> <amplitude> <phase_offset_deg> <emotion>
//
// Both arms oscillate around a center pose. Amplitude is a fraction of
// full travel (0..1). phase 0 = arms together, phase 180 = opposite.
//
// Pinout matches the working stepper_test.ino (ESP32):
//   Left  ULN2003 IN1..IN4 -> GPIO 13, 12, 14, 27
//   Right ULN2003 IN1..IN4 -> GPIO 26, 25, 33, 32
// Power the ULN2003 boards from 5 V; common GND with the ESP32.

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

// Full revolution = 4096 half-steps. Amplitude 1.0 swings +/- this many steps from center.
const int MAX_SWING_STEPS = 1024;   // 1/4 turn each side
const int STEP_DELAY_MS   = 2;      // raise to 3-5 if motors buzz or skip

int leftPhase  = 0;
int rightPhase = 0;
int leftPos    = 0;   // current step count relative to center
int rightPos   = 0;

float periodS   = 1.40;
float amplitude = 0.40;
float phaseDeg  = 0.0;
String emotion  = "neutral";

void applyPhase(const int pins[4], int phase) {
  for (int i = 0; i < 4; i++) digitalWrite(pins[i], SEQ[phase][i]);
}

void releaseCoils(const int pins[4]) {
  for (int i = 0; i < 4; i++) digitalWrite(pins[i], LOW);
}

void stepLeft(int dir) {
  leftPhase = (leftPhase + dir + 8) % 8;
  applyPhase(LEFT_PINS, leftPhase);
  leftPos  += dir;
}

void stepRight(int dir) {
  rightPhase = (rightPhase + dir + 8) % 8;
  applyPhase(RIGHT_PINS, rightPhase);
  rightPos += dir;
}

// Step one motor toward a target position (one step max per call).
void stepToward(void (*stepFn)(int), int &pos, int target) {
  if (pos < target) stepFn(+1);
  else if (pos > target) stepFn(-1);
}

void applyLine(String line) {
  if (!line.startsWith("ARM ")) return;
  float p, a, ph;
  char emo[16] = "neutral";
  if (sscanf(line.c_str(), "ARM %f %f %f %15s", &p, &a, &ph, emo) == 4) {
    periodS   = p > 0.05f ? p : 0.05f;
    amplitude = constrain(a, 0.0f, 1.0f);
    phaseDeg  = ph;
    emotion   = emo;
  }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) {
    pinMode(LEFT_PINS[i], OUTPUT);
    pinMode(RIGHT_PINS[i], OUTPUT);
  }
  releaseCoils(LEFT_PINS);
  releaseCoils(RIGHT_PINS);
  Serial.println("Stepper arms ready");
}

void loop() {
  // Non-blocking serial: read any complete lines that arrived.
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length()) applyLine(line);
  }

  // Desired positions from the same sine model the servo firmware used.
  float t     = millis() / 1000.0f;
  float w     = TWO_PI / periodS;
  float phase = phaseDeg * DEG_TO_RAD;
  int swing   = (int)(amplitude * MAX_SWING_STEPS);

  int leftTarget  = (int)(swing * sin(w * t));
  int rightTarget = (int)(swing * sin(w * t + phase));

  // Move at most one step per arm per loop so both stay roughly in sync.
  stepToward(stepLeft,  leftPos,  leftTarget);
  stepToward(stepRight, rightPos, rightTarget);

  // Small delay sets the maximum step rate (matches stepper_test).
  delay(STEP_DELAY_MS);
}