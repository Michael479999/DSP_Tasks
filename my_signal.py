# -------------------------
# Signal representation
# -------------------------
import numpy as np


class Signal:
    """
    Represents a discrete-time signal as a numpy array with a start index n0.
    If samples correspond to indices n0 ... n0 + len(values) - 1.
    """
    def __init__(self, values: np.ndarray, start_index: int, name: str = "signal"):
        self.values = np.asarray(values, dtype=float)
        self.start = int(start_index)
        self.name = name

    @classmethod
    def from_dict(cls, samples_dict, name="signal"):
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
        return cls(arr, start, name)

    @classmethod
    def from_file(cls, filename, name=None):
        """
        Read file with format:
        First line: N (number of samples)
        Next N lines: index value
        """
        samples = {}
        with open(filename, 'r') as f:
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
        if name is None:
            import os
            name = os.path.splitext(os.path.basename(filename))[0]
        return cls.from_dict(samples, name=name)

    def indices(self):
        if self.values.size == 0:
            return np.array([], dtype=int)
        return np.arange(self.start, self.start + self.values.size)

    def copy(self, name=None):
        return Signal(self.values.copy(), self.start, name if name else self.name + "_copy")

    def scaled(self, factor, name=None):
        return Signal(self.values * factor, self.start, name if name else f"{self.name}_x{factor}")
    


    
    def shifted(self, k, name=None):
        # x(n - k) => advance by k (start increases by k), x(n + k) => delay by k (start decreases)
        # We'll implement shifting semantics so that calling shifted(k) returns x(n - k)
        # where k positive means advance (the samples move left in n).
        new_start = self.start - k
        return Signal(self.values.copy(), new_start, name if name else f"{self.name}_shift_{k}")

    def folded(self, name=None):
        # x(-n): reverse samples and change start index
        if self.values.size == 0:
            return Signal(np.array([]), 0, name if name else f"{self.name}_fold")
        # original indices: start ... start+L-1
        L = self.values.size
        new_values = self.values[::-1].copy()
        # new start = - (old_end)
        old_end = self.start + L - 1
        new_start = -old_end
        return Signal(new_values, new_start, name if name else f"{self.name}_fold")

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

    @staticmethod
    def add(signals, name="sum"):
        """Add a list of signals and return a new Signal aligned correctly."""
        if not signals:
            return Signal(np.array([]), 0, name)
        # compute global start and end
        starts = [s.start for s in signals if s.values.size > 0]
        ends = [s.start + s.values.size - 1 for s in signals if s.values.size > 0]
        if not starts:
            return Signal(np.array([]), 0, name)
        full_start = min(starts)
        full_end = max(ends)
        L = full_end - full_start + 1
        total = np.zeros(L, dtype=float)
        for s in signals:
            if s.values.size == 0:
                continue
            idx = s.start - full_start
            total[idx:idx + s.values.size] += s.values
        return Signal(total, full_start, name)

    @staticmethod
    def subtract(sig_a, sig_b, name=None):
        """Compute sig_a - sig_b"""
        if name is None:
            name = f"{sig_a.name}_minus_{sig_b.name}"
        a, b, full_start = Signal._align(sig_a, sig_b)
        return Signal(a - b, full_start, name)

    def __str__(self):
        return f"{self.name}: start={self.start}, len={self.values.size}"
