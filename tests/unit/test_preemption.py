from ict_agent.runtime import preemption


def test_preemption_flags_roundtrip():
    preemption.set_autonomous_turn(True)
    assert preemption.is_autonomous_turn() is True

    preemption.request_preempt()
    assert preemption.is_preempt_requested() is True

    preemption.clear_preempt_request()
    assert preemption.is_preempt_requested() is False

    preemption.set_shell_interrupt_on_preempt(True)
    assert preemption.shell_interrupt_on_preempt() is True

    preemption.set_autonomous_turn(False)
    preemption.set_shell_interrupt_on_preempt(False)
