"""Pure validation gates for temporal forecasts."""


def deployment_eligibility(candidate_water, candidate_classes, baseline_water, baseline_classes):
    """Require a candidate to match or beat persistence on both forecast tasks."""
    water_delta = float(candidate_water["macro_f1"] - baseline_water["macro_f1"])
    class_delta = float(candidate_classes["macro_f1"] - baseline_classes["macro_f1"])
    eligible = water_delta >= 0.0 and class_delta >= 0.0
    return {
        "eligible": bool(eligible),
        "water_macro_f1_delta_vs_persistence": water_delta,
        "relative_class_macro_f1_delta_vs_persistence": class_delta,
        "rule": "Candidate macro-F1 must be no worse than persistence on both water extent and relative classes.",
    }
