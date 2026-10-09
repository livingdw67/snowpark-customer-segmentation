import re

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from segmentation.features import RULE
from segmentation.outcome_guided import OutcomeGuidedSegmenter, leaf_rules


def rule_holds(rule: str, row: pd.Series) -> bool:
    """Evaluate a rule string produced by leaf_rules against one customer."""
    for part in rule.split(" AND "):
        if part == "UK":
            ok = row["is_uk"] > 0.5
        elif part == "outside UK":
            ok = row["is_uk"] <= 0.5
        elif m := re.fullmatch(r"(\S+) < (\w+) <= (\S+)", part):
            ok = float(m[1]) < row[m[2]] <= float(m[3])
        elif m := re.fullmatch(r"(\w+) > (\S+)", part):
            ok = row[m[1]] > float(m[2])
        elif m := re.fullmatch(r"(\w+) <= (\S+)", part):
            ok = row[m[1]] <= float(m[2])
        else:
            raise ValueError(f"unparsed rule part: {part}")
        if not ok:
            return False
    return True


@pytest.fixture(scope="module")
def fitted(pooled_customers):
    return OutcomeGuidedSegmenter(n_segments=4).fit(pooled_customers)


def test_leaf_rules_describe_every_customer_correctly(customers):
    tree = DecisionTreeRegressor(max_leaf_nodes=8, random_state=0)
    tree.fit(customers[RULE], np.log1p(customers["net_revenue"]))
    rules = leaf_rules(tree, RULE)
    leaves = tree.apply(customers[RULE])
    # Thresholds are rounded to 2 decimals, so allow customers within rounding of a cut.
    wrong = sum(not rule_holds(rules[leaf], row) for leaf, (_, row) in zip(leaves, customers.iterrows()))
    assert wrong / len(customers) < 0.01


def test_leaf_rules_merge_repeated_splits(customers):
    tree = DecisionTreeRegressor(max_depth=4, random_state=0)
    tree.fit(customers[["net_revenue"]], customers["net_revenue"])
    for rule in leaf_rules(tree, ["net_revenue"]).values():
        assert rule.count("net_revenue") == 1


def test_segments_are_ranked_by_predicted_value(fitted, customers):
    out = fitted.predict(customers)
    assert set(out.segment) <= {f"C{i}" for i in range(1, 5)}
    means = out.groupby("segment")["expected_revenue"].mean()
    assert means.idxmax() == "C1"
    assert means.is_monotonic_decreasing


def test_scores_are_valid(fitted, customers):
    out = fitted.predict(customers)
    assert out.p_active.between(0, 1).all()
    assert (out.expected_revenue >= 0).all()


def test_model_ranks_customers_by_latent_value(fitted, customers):
    from sklearn.metrics import roc_auc_score
    out = fitted.predict(customers)
    assert roc_auc_score(customers["next_active"], out.p_active) > 0.7


def test_every_segment_has_a_rule(fitted):
    assert set(fitted.rules_) == set(fitted.leaf_order_.values())
    assert fitted.profiles_["rule"].notna().all()
