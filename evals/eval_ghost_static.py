from environments.ghost_trace.ghost_trace import load_environment as original_load_environment

def load_environment(**kwargs):
    """
    Static wrapper for Ghost Trace environment.
    Forces a fixed seed and number of examples to create a consistent evaluation set.
    """
    kwargs["seed"] = 100
    kwargs["num_examples"] = 50
    return original_load_environment(**kwargs)
