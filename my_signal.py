# -------------------------
# Signal representation
# -------------------------
import os
from typing import Any, Callable, List, Optional, Union
import numpy as np


class Signal:
    """
    Represents a discrete-time signal as a numpy array with a start index n0.
    If samples correspond to indices n0 ... n0 + len(values) - 1.
    """
    def __init__(self, values: np.ndarray, start_index: int, path: str, name: str = "signal"):
        self.values = np.asarray(values, dtype=float)
        self.start = int(start_index)
        self.name = name
        self.path = path

    @classmethod
    def from_dict(cls, samples_dict, path, name="signal"):
        """Create a Signal from dict {index: value}. Indices needn't be contiguous."""
        if not samples_dict:
            return cls(np.array([]), 0, name)
        indices = sorted(samples_dict.keys())
        start = indices[0]
        end = indices[-1]
        length = end - start + 1
        arr = np.zeros(length, dtype=float)
        for i in indices:
            arr[i - start] = samples_dict[i]
        return cls(arr, start, path, name)

    @classmethod
    def from_files(cls, filenames: List[str], callback: Callable[[Union["Signal", Exception]], None]) -> List["Signal"]:
        signals = []
        for f in filenames:
            try:
                sig = cls.from_file(f)
            except Exception as e:
                callback(e)
            else:
                callback(sig)
                signals.append(sig)
        return signals
    
    @classmethod
    def from_file(cls, filepath: str) -> "Signal":
        """
        Read file with format:
        First line: N (number of samples)
        Next N lines: index value
        """
        samples = {}
        with open(filepath, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip() != ""]
        if not lines:
            raise ValueError("Empty file")
        try:
            N = int(lines[0])
        except:
            raise ValueError("First line must be integer N (number of samples)")
        if len(lines) - 1 < N:
            raise ValueError(f"File says {N} samples but only {len(lines) - 1} data lines found.")
        for i in range(N):
            line = lines[1 + i]
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Line {i+2} is not 'index value'")
            idx = int(parts[0])
            val = float(parts[1])
            samples[idx] = val

        return cls.from_dict(samples, filepath, name=os.path.splitext(os.path.basename(filepath))[0])

    def indices(self):
        if self.values.size == 0:
            return np.array([], dtype=int)
        return np.arange(self.start, self.start + self.values.size)

    def copy(self, name=None):
        new_name = name if name else self.name
        return Signal(self.values.copy(), self.start, self.rename(new_name), new_name)

    def scaled(self, factor, name=None):
        new_name = name if name else f"{self.name}_x{factor}"
        return Signal(self.values * factor, self.start, self.rename(new_name), new_name)
    
    def shifted(self, k, name=None):
        # x(n - k) => advance by k (start increases by k), x(n + k) => delay by k (start decreases)
        # We'll implement shifting semantics so that calling shifted(k) returns x(n - k)
        # where k positive means advance (the samples move left in n).
        new_start = self.start - k
        new_name = name if name else f"{self.name}_shift_{k}"
        return Signal(self.values.copy(), new_start, self.rename(new_name), new_name)

    def folded(self, name=None):
        # x(-n): reverse samples and change start index
        new_name = name if name else f"{self.name}_fold"
        if self.values.size == 0:
            return Signal(np.array([]), 0, self.rename(new_name), new_name)
        # original indices: start ... start+L-1
        L = self.values.size
        new_values = self.values[::-1].copy()
        # new start = - (old_end)
        old_end = self.start + L - 1
        new_start = -old_end
        return Signal(new_values, new_start, self.rename(new_name), new_name)

    @staticmethod
    def _align(sig1, sig2):
        """
        Align two signals and return arrays (arr1, arr2, common_start)
        """
        if sig1.values.size == 0 and sig2.values.size == 0:
            return np.array([]), np.array([]), 0
        # Determine full span
        starts = []
        ends = []
        for s in (sig1, sig2):
            if s.values.size > 0:
                starts.append(s.start)
                ends.append(s.start + s.values.size - 1)
        if not starts:
            return np.array([]), np.array([]), 0
        full_start = min(starts)
        full_end = max(ends)
        L = full_end - full_start + 1
        a1 = np.zeros(L, dtype=float)
        a2 = np.zeros(L, dtype=float)
        if sig1.values.size > 0:
            s = sig1.start - full_start
            a1[s:s + sig1.values.size] = sig1.values
        if sig2.values.size > 0:
            s = sig2.start - full_start
            a2[s:s + sig2.values.size] = sig2.values
        return a1, a2, full_start

    def add(self, signals: List["Signal"], name=None):
        """Add the current signal to one or more signals and return a new aligned Signal."""
        all_signals = [self, *signals]

        if not name:
            name = "_plus_".join(list(map(lambda x: x.name, all_signals)))

        if not all_signals:
            return Signal(np.array([]), 0, self.rename(name), name)

        # Compute global start and end
        starts = [s.start for s in all_signals if s.values.size > 0]
        ends = [s.start + s.values.size - 1 for s in all_signals if s.values.size > 0]
        if not starts:
            return Signal(np.array([]), 0, self.rename(name), name)

        full_start = min(starts)
        full_end = max(ends)
        L = full_end - full_start + 1
        total = np.zeros(L, dtype=float)

        for s in all_signals:
            if s.values.size == 0:
                continue
            idx = s.start - full_start
            total[idx:idx + s.values.size] += s.values

        return Signal(total, full_start, self.rename(name), name)

    def subtract(self, other: "Signal", name=None):
        """Subtract another signal from the current one and return a new aligned Signal."""
        if name is None:
            name = f"{self.name}_minus_{other.name}"

        a, b, full_start = self._align(self, other)
        return Signal(a - b, full_start, self.rename(name), name)

    def __str__(self):
        return f"{self.name}: start={self.start}, len={self.values.size}"
    
    def rename(self, new_name: str):
        """Rename the file at self.path to the given new_name (in the same directory)."""
        if not hasattr(self, "path") or not self.path:
            raise ValueError("Signal has no valid path to rename.")

        dir_path = os.path.dirname(self.path)
        new_path = os.path.join(dir_path, new_name)

        _, ext = os.path.splitext(self.path)
        if not os.path.splitext(new_name)[1]:
            new_path += ext

        return new_path
