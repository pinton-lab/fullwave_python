from fullwave_simulation.conditions import Condition


def test_instance():
    condition = Condition()
    assert isinstance(condition, Condition)
