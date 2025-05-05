import os

if __name__ == "__main__":
    folder = os.path.join("src", "config", "yaml", "experiments")
    print(len(os.listdir(folder)))
