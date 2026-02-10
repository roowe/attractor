# tests/test_conditions.py
from attractor.conditions import evaluate_condition
from attractor.models import Context, Outcome, StageStatus


def test_evaluate_empty_condition():
    assert evaluate_condition("", None, None) == True


def test_evaluate_outcome_equals():
    outcome = Outcome(status=StageStatus.SUCCESS)
    assert evaluate_condition("outcome=success", outcome, None) == True
    assert evaluate_condition("outcome=fail", outcome, None) == False


def test_evaluate_outcome_not_equals():
    outcome = Outcome(status=StageStatus.SUCCESS)
    assert evaluate_condition("outcome!=fail", outcome, None) == True
    assert evaluate_condition("outcome!=success", outcome, None) == False


def test_evaluate_context_value():
    context = Context()
    context.set("tests_passed", "true")
    assert evaluate_condition("context.tests_passed=true", None, context) == True


def test_evaluate_and_clause():
    outcome = Outcome(status=StageStatus.SUCCESS)
    context = Context()
    context.set("flag", "true")
    assert evaluate_condition("outcome=success && context.flag=true", outcome, context) == True


def test_evaluate_and_clause_fail():
    outcome = Outcome(status=StageStatus.SUCCESS)
    context = Context()
    context.set("flag", "false")
    assert evaluate_condition("outcome=success && context.flag=true", outcome, context) == False
