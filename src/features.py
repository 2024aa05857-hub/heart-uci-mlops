from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@dataclass
class FeatureSpec:
    numeric: list[str]

def build_preprocess(spec: FeatureSpec) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, spec.numeric)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
