from src.config.experiment import generate_experiments, save_experiment

if __name__ == "__main__":
    datasets = ["eesm19"]  # Datasets to generate experiments for.
    methods = [
        "Full_3",
        "Full_4",
        "Full_6",
        "LoRA20_3",
        "LoRA20_4",
        "LoRA20_6",
        "SCL20_3",
        "SCL20_4",
        "SCL20_6",
    ]  # Methods to generate experiments for (None = 'all').
    id = 99

    for exp in generate_experiments(datasets, methods, id):
        save_experiment(exp)
