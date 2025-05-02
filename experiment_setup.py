from src.config.experiment import generate_experiments, save_experiment

if __name__ == "__main__":
    datasets = ["eesm19"]  # Datasets to generate experiments for.
    methods = [
        "Full",
        "Fish",
        "LoRA20",
    ]  # Methods to generate experiments for (None = 'all').
    train_size = None
    folds = [1, 4]
    seed = 42

    for exp in generate_experiments(
        datasets=datasets,
        methods=methods,
        train_size=train_size,
        seed=seed,
        folds=folds,
    ):
        save_experiment(exp)
