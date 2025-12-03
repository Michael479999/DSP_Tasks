import os
from my_signal import Signal

script_dir = os.path.abspath(__file__)
base_dir = os.path.dirname(script_dir)

def get_file_path(filename: str, ext: str = "txt") -> str:
	return os.path.join(base_dir, f"{filename}.{ext}")

def save_signal(filename: str, sig: Signal, ext: str = "txt") -> None:
    path = get_file_path(filename)
    with open(path, "w") as f:
        f.write(f"{len(sig.values)}\n")
        for i, v in zip(sig.indices(), sig.values):
            if isinstance(v, complex):
                f.write(f"{i} {v.real} {v.imag}\n")
            else:
                f.write(f"{i} {v}\n")