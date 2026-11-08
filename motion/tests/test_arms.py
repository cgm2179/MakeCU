"""Run from the repo root: python3 -m unittest motion.tests.test_arms -v"""
import unittest

from motion.control.arm_params import motion_for_emotion
from motion.control.emotion_arms import EmotionArms
from motion.drivers.arm_link import ArmLink


class TestArms(unittest.TestCase):
    def test_happy_is_fast_large_and_out_of_phase(self):
        happy = motion_for_emotion("happy")
        sad = motion_for_emotion("sad")
        self.assertEqual(happy.phase_offset_deg, 180.0)
        self.assertEqual(sad.phase_offset_deg, 0.0)
        self.assertLess(happy.period_s, sad.period_s)
        self.assertGreater(happy.amplitude, sad.amplitude)

    def test_apply_sends_a_line_the_arduino_can_parse(self):
        link = ArmLink()
        profile = EmotionArms(link).apply("sad")
        self.assertEqual(profile.phase_offset_deg, 0.0)
        self.assertTrue(link.last_line.startswith("ARM "))
        parts = link.last_line.split()
        self.assertEqual(parts[0], "ARM")
        self.assertEqual(parts[4], "sad")
        self.assertAlmostEqual(float(parts[1]), profile.period_s)
        self.assertAlmostEqual(float(parts[3]), 0.0)


if __name__ == "__main__":
    unittest.main()
