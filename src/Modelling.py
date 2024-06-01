import pandas as pd
import polars as pl
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)

df = pd.read_csv('../data/frenchmtpl_clean.csv', sep=';')

df["Frequency"] = df["ClaimNb"] / df["Exposure"]

df_train, df_test = train_test_split(df, test_size=0.33, random_state=0)

log_scale_transformer = make_pipeline(
    FunctionTransformer(np.log, validate=False), StandardScaler()
)

linear_model_preprocessor = ColumnTransformer(
    [
        ("passthrough_numeric", "passthrough", ["BonusMalus"]),
        (
            "binned_numeric",
            KBinsDiscretizer(n_bins=10, random_state=0),
            ["VehAge", "DrivAge"],
        ),
        ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        (
            "onehot_categorical",
            OneHotEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
        ),
    ],
    remainder="drop",
)


poisson_glm = Pipeline(
    [
        ("preprocessor", linear_model_preprocessor),
        ("regressor", PoissonRegressor(alpha=1e-12, solver="newton-cholesky")),
    ]
)


model_sklearn = poisson_glm.fit(
    df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"]
)



exit()
def score_estimator(estimator, test_set):
    """Score an estimator on the test set."""
    y_pred = estimator.predict(test_set)

    print(
        "MSE: %.3f"
        % mean_squared_error(
            test_set["Frequency"], y_pred, sample_weight=test_set["Exposure"]
        )
    )
    print(
        "MAE: %.3f"
        % mean_absolute_error(
            test_set["Frequency"], y_pred, sample_weight=test_set["Exposure"]
        )
    )

    # Ignore non-positive predictions, as they are invalid for
    # the Poisson deviance.
    mask = y_pred > 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(
            "WARNING: Estimator yields invalid, non-positive predictions "
            f" for {n_masked} samples out of {n_samples}. These predictions "
            "are ignored when computing the Poisson deviance."
        )

    print(
        "mean Poisson deviance: %.3f"
        % mean_poisson_deviance(
            test_set["Frequency"][mask],
            y_pred[mask],
            sample_weight=test_set["Exposure"][mask],
        )
    )


print("PoissonRegressor evaluation:")
score_estimator(poisson_glm, df_test)
