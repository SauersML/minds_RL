from environments.self_prediction.self_prediction import load_environment as original_load_environment

def load_environment(**kwargs):
    """
    Static wrapper for Self-Prediction environment.
    Forces a fixed seed and number of examples to create a consistent evaluation set.
    """
    kwargs["seed"] = 42
    kwargs["num_examples"] = 50
    return original_load_environment(**kwargs)
