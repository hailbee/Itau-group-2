import numpy as np

def extract_misclassified(results_df, threshold):
    """
    Add predicted_label + error_type columns and return only misclassified rows.

    error_type:
      - false_negative: label=1, pred=0
      - false_positive: label=0, pred=1
    """
    df = results_df.copy()

    df["label"] = df["label"].astype(int)
    df["predicted_label"] = (df["max_similarity"] > threshold).astype(int)

    df["error_type"] = np.where(
        (df["label"] == 1) & (df["predicted_label"] == 0),
        "false_negative",
        np.where(
            (df["label"] == 0) & (df["predicted_label"] == 1),
            "false_positive",
            "correct"
        )
    )

    return df[df["error_type"] != "correct"].reset_index(drop=True)
