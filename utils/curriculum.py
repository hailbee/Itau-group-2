import numpy as np

def get_curriculum_ratios(epoch, epochs):
    t = epoch / max(epochs - 1, 1)

    if t < 0.10:
        return {"easy": 0.60, "medium": 0.40, "hard": 0.00}
    elif t < 0.70:
        return {"easy": 0.10, "medium": 0.80, "hard": 0.10}
    else:
        return {"easy": 0.00, "medium": 0.70, "hard": 0.30}
