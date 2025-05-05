from src.config.experiment import generate_experiments, save_experiment

if __name__ == "__main__":
    datasets = [
        "eesm19",
        "isruc_sg2",
        "isruc_sg3",
    ]  # Datasets to generate experiments for.
    methods = None  # Methods to generate experiments for (None = 'all').
    train_size = None
    folds = None
    seed = 42
    key = None

    for exp in generate_experiments(
        datasets=datasets,
        methods=methods,
        train_size=train_size,
        seed=seed,
        folds=folds,
        key=key,
    ):
        save_experiment(exp)
