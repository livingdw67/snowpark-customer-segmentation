from segmentation.profiles import describe, z_space


def test_z_space_is_standardized(customers):
    Z = z_space(customers)
    assert (Z.mean().abs() < 1e-9).all()


def test_traits_name_what_sets_a_segment_apart(customers):
    labels = (customers["q4_revenue_share"] > 0.6).map({True: "holiday", False: "rest"})
    out = describe(customers, labels).set_index("segment")
    assert "holiday-heavy" in out.loc["holiday", "traits"]
    assert out["customers"].sum() == len(customers)
    assert abs(out["share"].sum() - 1) < 1e-9


def test_recency_reads_the_right_way_round(customers):
    labels = (customers["recency_days"] > customers["recency_days"].quantile(0.8)).map(
        {True: "old", False: "rest"})
    assert "lapsed" in describe(customers, labels).set_index("segment").loc["old", "traits"]
