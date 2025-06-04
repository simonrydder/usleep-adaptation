import h5py


def print_hdf5_structure(file_path: str) -> None:
    def print_attrs(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}[Dataset] {name} - shape: {obj.shape}, dtype: {obj.dtype}")

    with h5py.File(file_path, "r") as f:
        f.visititems(print_attrs)


if __name__ == "__main__":
    print_hdf5_structure("data/hdf5/svuh.hdf5")
