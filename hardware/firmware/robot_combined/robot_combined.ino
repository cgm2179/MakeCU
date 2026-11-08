// Combined ESP32 firmware: obstacle-triggered DC drive plus emotion-driven
// 28BYJ-48 stepper arms. The Python side sends ARM lines over USB serial.
//
// Stepper wiring (ESP32 GPIO -> ULN2003 IN1..IN4):
//   Left:  13, 12, 14, 27
//   Right: 26, 25, 33, 32
//
// Sensor and DC motor shield pins are the ones from obstacle_drive.ino.
// Power motors from suitable external supplies and connect grounds to ESP32 GND.
// Use a level shifter/resistor divider if the ultrasonic ECHO output is 5 V.

#include <math.h>
#include <stdio.h>
#include <string.h>

const int TRIG = 5;
const int ECHO = 18;

const int IN1 = 15;
const int IN2 = 19;
const int ENA = 4;
const int IN3 = 21;
const int IN4 = 22;
const int ENB = 2;
const int DETECT_CM = 30;

const int LEFT_PINS[4] = {13, 12, 14, 27};
const int RIGHT_PINS[4] = {26, 25, 33, 32};

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

const int MAX_SWING_STEPS = 1024;
const uint32_t STEP_INTERVAL_US = 2000;
const uint32_t SENSOR_INTERVAL_US = 60000;
const uint32_t ECHO_TIMEOUT_US = 30000;

int leftPhase = 0;
int rightPhase = 0;
int leftPos = 0;
int rightPos = 0;
float periodS = 1.40f;
float amplitude = 0.40f;
float phaseDeg = 0.0f;
char emotion[16] = "neutral";

enum SensorState { SENSOR_IDLE, TRIGGER_HIGH, WAIT_ECHO_RISE, WAIT_ECHO_FALL };
SensorState sensorState = SENSOR_IDLE;
uint32_t triggerStartedUs = 0;
uint32_t echoStartedUs = 0;
uint32_t nextTriggerUs = 0;
uint32_t nextStepUs = 0;

void applyPhase(const int pins[4], int phase) {
  for (int i = 0; i < 4; i++) digitalWrite(pins[i], SEQ[phase][i]);
}

void stepLeft(int direction) {
  leftPhase = (leftPhase + direction + 8) % 8;
  applyPhase(LEFT_PINS, leftPhase);
  leftPos += direction;
}

void stepRight(int direction) {
  rightPhase = (rightPhase + direction + 8) % 8;
  applyPhase(RIGHT_PINS, rightPhase);
  rightPos += direction;
}

void stepToward(void (*stepFn)(int), int &position, int target) {
  if (position < target) stepFn(1);
  else if (position > target) stepFn(-1);
}

void updateDrive(long distanceCm) {
  if (distanceCm < DETECT_CM) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    digitalWrite(ENA, HIGH);
    digitalWrite(ENB, HIGH);
  } else {
    digitalWrite(ENA, LOW);
    digitalWrite(ENB, LOW);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
  }
}

void finishDistance(long distanceCm) {
  updateDrive(distanceCm);
  sensorState = SENSOR_IDLE;
}

void updateDistanceSensor() {
  uint32_t now = micros();
  switch (sensorState) {
    case SENSOR_IDLE:
      if ((int32_t)(now - nextTriggerUs) >= 0) {
        digitalWrite(TRIG, LOW);
        digitalWrite(TRIG, HIGH);
        triggerStartedUs = now;
        nextTriggerUs = now + SENSOR_INTERVAL_US;
        sensorState = TRIGGER_HIGH;
      }
      break;
    case TRIGGER_HIGH:
      if (now - triggerStartedUs >= 10) {
        digitalWrite(TRIG, LOW);
        echoStartedUs = now;
        sensorState = WAIT_ECHO_RISE;
      }
      break;
    case WAIT_ECHO_RISE:
      if (digitalRead(ECHO) == HIGH) {
        echoStartedUs = now;
        sensorState = WAIT_ECHO_FALL;
      } else if (now - echoStartedUs >= ECHO_TIMEOUT_US) {
        finishDistance(999);
      }
      break;
    case WAIT_ECHO_FALL:
      if (digitalRead(ECHO) == LOW) {
        long durationUs = now - echoStartedUs;
        finishDistance((long)(durationUs * 0.034f / 2.0f));
      } else if (now - echoStartedUs >= ECHO_TIMEOUT_US) {
        finishDistance(999);
      }
      break;
  }
}

void applyLine(const char *line) {
  float newPeriod, newAmplitude, newPhase;
  char newEmotion[16];
  if (sscanf(line, "ARM %f %f %f %15s",
             &newPeriod, &newAmplitude, &newPhase, newEmotion) == 4) {
    periodS = newPeriod > 0.05f ? newPeriod : 0.05f;
    amplitude = constrain(newAmplitude, 0.0f, 1.0f);
    phaseDeg = newPhase;
    strncpy(emotion, newEmotion, sizeof(emotion) - 1);
    emotion[sizeof(emotion) - 1] = '\0';
  }
}

void readSerialCommands() {
  static char line[96];
  static size_t length = 0;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      line[length] = '\0';
      if (length > 0) applyLine(line);
      length = 0;
    } else if (c != '\r') {
      if (length < sizeof(line) - 1) line[length++] = c;
      else length = 0;
    }
  }
}

void updateSteppers() {
  uint32_t now = micros();
  if ((int32_t)(now - nextStepUs) < 0) return;
  nextStepUs = now + STEP_INTERVAL_US;

  float t = millis() / 1000.0f;
  float angularSpeed = TWO_PI / periodS;
  float phase = phaseDeg * DEG_TO_RAD;
  int swing = (int)(amplitude * MAX_SWING_STEPS);
  int leftTarget = (int)(swing * sinf(angularSpeed * t));
  int rightTarget = (int)(swing * sinf(angularSpeed * t + phase));

  stepToward(stepLeft, leftPos, leftTarget);
  stepToward(stepRight, rightPos, rightTarget);
}

void setup() {
  Serial.begin(115200);

  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  digitalWrite(TRIG, LOW);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);
  updateDrive(999);

  for (int i = 0; i < 4; i++) {
    pinMode(LEFT_PINS[i], OUTPUT);
    pinMode(RIGHT_PINS[i], OUTPUT);
  }

  Serial.println("Combined obstacle drive and emotion stepper firmware ready");
}

void loop() {
  readSerialCommands();
  updateDistanceSensor();
  updateSteppers();
}
