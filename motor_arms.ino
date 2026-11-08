// Pin mapping for ULN2003 Stepper Motor Driver
const int IN1 = 13;
const int IN2 = 12;
const int IN3 = 14;
const int IN4 = 27;

// 4-Step sequence for the internal magnets
const int steps[4][4] = {
  {1, 0, 0, 0},
  {0, 1, 0, 0},
  {0, 0, 1, 0},
  {0, 0, 0, 1}
};

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
}

// Function to move the motor a set number of steps
void move_motor(int steps_to_move, int direction) {
  for (int x = 0; x < steps_to_move; x++) {
    for (int i = 0; i < 4; i++) {
      // If direction is 1, go forward. If -1, go backward.
      int step_index = (direction == 1) ? i : (3 - i);

      digitalWrite(IN1, steps[step_index][0]);
      digitalWrite(IN2, steps[step_index][1]);
      digitalWrite(IN3, steps[step_index][2]);
      digitalWrite(IN4, steps[step_index][3]);
      delay(4); // 4 milliseconds between steps (Safe speed)
    }
  }
}

void loop() {
  // Move forward 500 steps (Roughly a 90-degree turn with internal gears)
  move_motor(500, 1);
  delay(1000); // Wait 1 second

  // Move backward 500 steps
  move_motor(500, -1);
  delay(1000); // Wait 1 second

}
