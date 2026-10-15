def stop_at_step(state, control, target):
    if target <= 0:
        raise ValueError("Target must be positive")
    if state.global_step >= target:
        control.should_save = True
        control.should_training_stop = True
    return control
