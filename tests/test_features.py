import pandas as pd
from src.features import FeatureSpec, build_preprocess

def test_preprocess_shape():
    X = pd.DataFrame({"age":[50,60], "sex":[1,0]})
    pre = build_preprocess(FeatureSpec(numeric=["age","sex"]))
    Xt = pre.fit_transform(X)
    assert Xt.shape == (2,2)
