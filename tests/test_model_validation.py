import unittest

from src.model_validation import deployment_eligibility


class TestDeploymentEligibility(unittest.TestCase):
    def test_rejects_candidate_worse_than_persistence(self):
        result = deployment_eligibility(
            {"macro_f1": 0.43}, {"macro_f1": 0.14},
            {"macro_f1": 0.90}, {"macro_f1": 0.54},
        )
        self.assertFalse(result["eligible"])
        self.assertLess(result["water_macro_f1_delta_vs_persistence"], 0)
        self.assertLess(result["relative_class_macro_f1_delta_vs_persistence"], 0)

    def test_accepts_candidate_matching_or_beating_persistence(self):
        result = deployment_eligibility(
            {"macro_f1": 0.91}, {"macro_f1": 0.55},
            {"macro_f1": 0.90}, {"macro_f1": 0.54},
        )
        self.assertTrue(result["eligible"])


if __name__ == "__main__":
    unittest.main()
