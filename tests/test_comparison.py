import pytest

from factories import RAW_OUTPUT, task
from sim2real.contracts import DifferenceKind, ScientificTaskSpec, compare_tasks


def kinds(report):
    return {d.variable: d.kind for d in report.differences}


def test_xc_comparison_varying_only_the_declared_variable_is_controlled():
    report = compare_tasks([task(xc="PBE"), task(xc="PBE0")], varied={"xc"})
    assert report.is_controlled
    assert kinds(report) == {"xc": DifferenceKind.DECLARED}
    assert report.unused_declarations == ()


def test_undeclared_intervention_difference_is_uncontrolled():
    report = compare_tasks([task(xc="PBE"), task(xc="PBE0")], varied=set())
    assert not report.is_controlled
    assert kinds(report) == {"xc": DifferenceKind.UNCONTROLLED}


def test_changed_physical_control_is_uncontrolled():
    report = compare_tasks([task(xc="PBE"), task(xc="PBE0", spin="collinear")], varied={"xc"})
    assert not report.is_controlled
    assert [d.variable for d in report.uncontrolled] == ["spin"]


def test_numerical_difference_is_allowed_but_needs_justification():
    report = compare_tasks([task(xc="PBE"), task(xc="PBE0", k_grid=(6, 6, 1))], varied={"xc"})
    assert report.is_controlled
    assert [d.variable for d in report.needs_numerical_justification] == ["k_grid"]
    assert report.needs_numerical_justification[0].values == ([4, 4, 1], [6, 6, 1])


def test_convergence_test_declares_the_numerical_variable():
    report = compare_tasks([task(k_grid=(4, 4, 1)), task(k_grid=(8, 8, 1))], varied={"k_grid"})
    assert kinds(report) == {"k_grid": DifferenceKind.DECLARED}
    assert report.needs_numerical_justification == ()


def test_fixed_and_relaxed_estimands_are_never_mixed():
    fixed = task(xc="PBE", estimand="E(atop) - E(fcc) at fixed geometries")
    relaxed = task(xc="PBE0", estimand="E(atop) - E(fcc), each relaxed with its own functional")
    report = compare_tasks([fixed, relaxed], varied={"xc", "estimand"})
    assert kinds(report)["estimand"] is DifferenceKind.UNCONTROLLED
    assert not report.is_controlled


def test_structure_change_must_be_declared():
    atop = task(structure=RAW_OUTPUT)
    hollow = task(structure=RAW_OUTPUT.model_copy(update={"sha256": "b" * 64}))
    assert not compare_tasks([atop, hollow], varied=set()).is_controlled
    assert compare_tasks([atop, hollow], varied={"structure"}).is_controlled


def test_inconsistent_classification_is_uncontrolled():
    a = task()
    b = ScientificTaskSpec.model_validate({
        **a.model_dump(), "interventions": {}, "controls": {**a.controls, "xc": "PBE"},
    })
    report = compare_tasks([a, b], varied={"xc"})
    assert [(d.variable, d.category) for d in report.uncontrolled] == [("xc", "mixed")]


def test_variable_missing_from_one_task_is_a_difference():
    a = task()
    b = ScientificTaskSpec.model_validate({**a.model_dump(), "controls": {"spin": "none"}})
    report = compare_tasks([a, b], varied=set())
    (difference,) = report.uncontrolled
    assert difference.variable == "coverage" and difference.values == ("0.25 ML", None)


def test_unused_declarations_are_reported():
    report = compare_tasks([task(xc="PBE"), task(xc="PBE0")], varied={"xc", "coverage"})
    assert report.unused_declarations == ("coverage",)


def test_comparison_needs_two_tasks():
    with pytest.raises(ValueError):
        compare_tasks([task()], varied={"xc"})
