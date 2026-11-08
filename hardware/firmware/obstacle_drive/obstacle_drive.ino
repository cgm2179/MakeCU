// Sensor pins
const int TRIG = 5;
const int ECHO = 18;

// Motor shield pins
const int IN1 = 15;  // motor A direction
const int IN2 = 19;
const int ENA = 4;   // motor A enable
const int IN3 = 21;  // motor B direction
const int IN4 = 22;
const int ENB = 2;   // motor B enable

const int DETECT_CM = 30;  // move if something is closer than this

long readDistanceCm() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  long duration = pulseIn(ECHO, HIGH, 30000);  // 30 ms timeout
  if (duration == 0) return 999;               // nothing detected
  return duration * 0.034 / 2;
}

void forward() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);  digitalWrite(IN4, LOW);  // flipped
  digitalWrite(ENA, HIGH); digitalWrite(ENB, HIGH);
}

void stopMotors() {
  digitalWrite(ENA, LOW); digitalWrite(ENB, LOW);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT); pinMode(ENA, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT); pinMode(ENB, OUTPUT);
  stopMotors();
}

void loop() {
  long d = readDistanceCm();
  Serial.println(d);

  if (d < DETECT_CM) forward();
  else stopMotors();

  delay(60);
}
