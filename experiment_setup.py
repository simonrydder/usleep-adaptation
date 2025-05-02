from src.config.experiment import generate_experiments, save_experiment

if __name__ == "__main__":
    datasets = ["eesm19"]  # Datasets to generate experiments for.
    methods = [
        "Full",
    ]  # Methods to generate experiments for (None = 'all').
    train_size = None
    folds = None
    seed = 42
    key = "Eo974696"

    for exp in generate_experiments(
        datasets=datasets,
        methods=methods,
        train_size=train_size,
        seed=seed,
        folds=folds,
        key=key,
    ):
        save_experiment(exp)
