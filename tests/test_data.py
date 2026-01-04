import pandas as pd
from src.data import load_and_clean

def test_target_binary(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({
        "age":[50], "sex":[1], "cp":[0], "trestbps":[120], "chol":[200],
        "fbs":[0], "restecg":[1], "thalach":[150], "exang":[0], "oldpeak":[1.0],
        "slope":[2], "ca":[0], "thal":[3], "target":[2]
    }).to_csv(p, index=False)
    df = load_and_clean(str(p))
    assert set(df["target"].unique()).issubset({0,1})
