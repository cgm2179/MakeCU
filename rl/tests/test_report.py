"""Run from the repo root: python3 -m unittest rl.tests.test_report -v"""
import sys
import tempfile
import unittest

from rl import eval as ev
from rl import report as rp
from rl.tests.test_eval import fake_items


class TestReport(unittest.TestCase):
    def test_test_runner_parses_counts_and_failures(self):
        ok = rp.run_unit_tests(runner=[sys.executable, "-c", "import sys; print('Ran 7 tests', file=sys.stderr)"])
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["ran"], 7)
        bad = rp.run_unit_tests(runner=[sys.executable, "-c",
                                        "import sys; print('FAIL: test_x (a.b)\\nRan 3 tests\\nFAILED (failures=1, skipped=2)', file=sys.stderr); sys.exit(1)"])
        self.assertFalse(bad["ok"])
        self.assertEqual((bad["ran"], bad["failed"], bad["skipped"]), (3, 1, 2))
        self.assertEqual(bad["failures"], ["FAIL: test_x (a.b)"])

    def test_report_contains_results_method_and_caveats(self):
        tests = rp.run_unit_tests(runner=[sys.executable, "-c", "import sys; print('Ran 2 tests', file=sys.stderr)"])
        text = rp.build_report(9, "Demo", fake_items(100, noise=0.05), ev.default_models(),
                               "**SYNTHETIC** test data", True, tests)
        for needle in ("Unit tests: ALL PASSED", "How well the algorithm performs", "bandit (from scratch)",
                       "AUC per fold", "Learning curve", "How the numbers were produced", "AUC = 3/4 = 0.75",
                       "labels are synthetic", "5-fold"):
            self.assertIn(needle, text)

    def test_failed_tests_are_shown_not_hidden(self):
        bad = rp.run_unit_tests(runner=[sys.executable, "-c", "import sys; print('Ran 1 tests', file=sys.stderr); sys.exit(1)"])
        self.assertIn("Unit tests: FAILURES", rp.test_section(bad))

    def test_write_report_names_file_by_step(self):
        with tempfile.TemporaryDirectory() as d:
            p = rp.write_report(4, "ratings", "hello", directory=d)
            self.assertEqual(p.name, "step04_ratings.md")
            self.assertEqual(p.read_text(), "hello")


if __name__ == "__main__":
    unittest.main()
