from src.config.experiment import generate_experiments, save_experiment

if __name__ == "__main__":
    datasets = ["eesm19"]  # Datasets to generate experiments for.
    methods = None  # Methods to generate experiments for (None = 'all').
    id = 1

    for exp in generate_experiments(datasets, methods, id):
        save_experiment(exp)
