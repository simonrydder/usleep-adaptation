from src.config.experiment import generate_experiments, save_experiment

if __name__ == "__main__":
    datasets = ["eesm19"]  # Datasets to generate experiments for.
    methods = [
        "Full",
        "Fish10",
        "LoRA20",
        "SCL20",
    ]  # Methods to generate experiments for (None = 'all').
    id = 91
    seed = 46

    for exp in generate_experiments(datasets, methods, id, seed):
        save_experiment(exp)
