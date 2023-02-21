
# Import libraries
import optuna

# define objective function to minimize


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -4.5, 4.5)
    y = trial.suggest_float("y", -4.5, 4.5)

    print(f"x: {x}, y: {y}, objective: {(1.5 - x + x*y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2}")

    return (1.5 - x + x*y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


def main() -> None:
    # create study object and optimize
    study: optuna.study.Study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=1000)
    # study.optimize(objective, n_trials=1000, catch=(TypeError, ValueError))
    study.optimize(objective, n_trials=1000, catch=Exception)

    # print results
    print(f"Best objective value: {study.best_value}")
    print(f"Best parameter: {study.best_params}")


if __name__ == "__main__":
    main()
