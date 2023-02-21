import optuna
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data = fetch_openml(name="adult", version=1, as_frame=True)
X = pd.get_dummies(data.data)
y = (data.target == ">50K") * 1


def objective(trial: optuna.trial.Trial) -> float:
    clf = RandomForestClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 32, log=True),
        min_samples_split=trial.suggest_float("min_samples_split", 0.0, 1.0),
    )

    score = cross_val_score(clf, X, y, cv=3)
    acucuracy = score.mean()
    return acucuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print(f"Best objective value: {study.best_value}")
print(f"Best parameter: {study.best_params}")
