import ast
import logging
from pathlib import Path

import pytest


@pytest.mark.parametrize("level", [logging.WARNING, logging.DEBUG])
def test_fit_tracking_does_not_shadow_debug_logger(level):
    source = Path(__file__).resolve().parents[3] / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    fit = next(node for node in trainer.body
               if isinstance(node, ast.FunctionDef) and node.name == "fit")
    assignment = next(node for node in ast.walk(fit)
                      if isinstance(node, ast.Assign)
                      and isinstance(node.value, ast.Call)
                      and isinstance(node.value.func, ast.Name)
                      and node.value.func.id == "Tracking")
    tracker_name = assignment.targets[0].id
    logger = logging.Logger("logging-scope-test", level=level)
    scope = {"logger": logger, "logging": logging, tracker_name: object()}
    checks = [node for node in ast.walk(fit)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr == "isEnabledFor"]
    assert checks
    for check in checks:
        expression = ast.Expression(body=check)
        assert eval(compile(expression, str(source), "eval"), scope) == (
            level == logging.DEBUG)
    metric_calls = [node for node in ast.walk(fit)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "log"
                    and isinstance(node.func.value, ast.Name)]
    assert len(metric_calls) == 2
    assert all(node.func.value.id == tracker_name for node in metric_calls)
