# -------------------------
# Signal representation
# -------------------------
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import math

def _hanning(N: int, n: int) -> np.ndarray:
    return 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(-n, n + 1) / N)

def _hamming(N: int, n: int) -> np.ndarray:
    return 0.54 + 0.46 * np.cos(2 * np.pi * np.arange(-n, n + 1) / N)

def _blackman(N: int, n: int) -> np.ndarray:
    factor = np.pi * np.arange(-n, n + 1) / N - 1
    return 0.42 + 0.5 * np.cos(2 * factor) + 0.08 * np.cos(4 * factor)

def _pass_filter(n: int, fc: float, zero_value: Optional[float] = None):
    if zero_value is None:
        zero_value = 2 * fc
        
    n_arr = np.arange(-n, n + 1) # [-3, -2, -1, 0, 1, 2, 3] for n=3
    omega = 2 * np.pi * fc
    values = 2 * fc * np.sin(omega * n_arr) / (n_arr * omega)
    return np.where(n_arr == 0, zero_value, values)

def _lowpass(n: int, normalized_cutoff_freqs: List[float]):
    fc = normalized_cutoff_freqs[0]
    return _pass_filter(n, fc)

def _highpass(n: int, normalized_cutoff_freqs: List[float]):
    fc = normalized_cutoff_freqs[0]
    filter = -_pass_filter(n, fc, zero_value=1 - 2 * fc)
    return filter

def _bandpass(n: int, normalized_cutoff_freqs: List[float]):
    fc1, fc2 = sorted(normalized_cutoff_freqs)
    zero_value = 2 * (fc2 - fc1)
    filter = _pass_filter(n, fc2, zero_value) - _pass_filter(n, fc1, zero_value)
    return filter

def _bandstop(n: int, normalized_cutoff_freqs: List[float]):
    fc1, fc2 = sorted(normalized_cutoff_freqs)
    zero_value = 1 - 2 * (fc2 - fc1)
    filter = _pass_filter(n, fc1, zero_value) - _pass_filter(n, fc2, zero_value)
    return filter


window_functions: Dict[str, Tuple[float, Callable[[int, int], np.ndarray]]] = {
    "rectangular": (0.9, lambda N: np.ones(N)),
    "hanning": (3.1, _hanning),
    "hamming": (3.3, _hamming),
    "blackman": (5.5, _blackman),
}

filters: Dict[str, Callable[[int, List[float]], np.ndarray]] = {
    "Low-pass": _lowpass,
    "High-pass": _highpass,
    "Band-pass": _bandpass,
    "Band-stop": _bandstop,
}

class Signal:
    """
    Represents a discrete-time signal as a numpy array with a start index n0.
    If samples correspond to indices n0 ... n0 + len(values) - 1.
    """
    
    def __init__(self, values: np.ndarray, start_index: int, name: str = "signal", is_freq_domain: bool = False):
        self.values = values
        self.start = int(start_index)
        self.name = name
        self.is_frequency_domain = is_freq_domain

    @classmethod
    def from_dict(cls, samples_dict, is_freq_domain: bool = False, name="signal"):
        """Create a Signal from dict {index: value}. Indices needn't be contiguous."""
        if not samples_dict:
            return cls(np.array([]), 0, name)
        indices = sorted(samples_dict.keys())
        start = indices[0]
        end = indices[-1]
        length = end - start + 1
        arr = np.zeros(length, dtype=float if not is_freq_domain else complex)
        for i in indices:
            arr[i - start] = samples_dict[i]
        return cls(arr, start, name, is_freq_domain)

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
    def from_templates(cls, filenames: List[str], callback: Callable[[Union["Signal", Exception]], None]) -> List["Signal"]:
        signals = []
        for f in filenames:
            try:
                sig = cls.from_template(f)
            except Exception as e:
                callback(e)
            else:
                callback(sig)
                signals.append(sig)
        return signals
    
    @classmethod
    def from_template(cls, filepath: str) -> "Signal":
        """
        Read a signal template file with format:
        N lines: 
            value
        """
        samples = {}

        with open(filepath, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip() != ""]

        if not lines:
            raise ValueError("Empty file")

        N = len(lines)
        
        for i in range(N):
            parts = lines[i].strip().split()

            if len(parts) != 1:
                raise ValueError(f"Line {i+1} is not valid: needs 'value'.")
            
            samples[i] = float(parts[0])

        return cls.from_dict(samples, is_freq_domain=False, name=os.path.splitext(os.path.basename(filepath))[0])
    
    @classmethod
    def from_file(cls, filepath: str) -> "Signal":
        """
        Read file with format:
        First line: N (number of samples)
        Next N lines:
            (index value) OR (index real imaginary)
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
            raise ValueError(f"File says {N} samples but only {len(lines)-1} data lines found.")
        
        is_freq_domain = False
        
        for i in range(N):
            line = lines[1 + i]
            parts = line.split()

            if len(parts) < 2:
                raise ValueError(f"Line {i+2} is not valid: needs at least 'index value'.")

            idx = int(parts[0])

            if len(parts) == 2:
                val = float(parts[1])
            elif len(parts) == 3:
                is_freq_domain = True
                val = complex(float(parts[1]), float(parts[2]))
            else:
                raise ValueError(f"Line {i+2} must be 'i v' or 'i real imag'.")

            samples[idx] = val

        return cls.from_dict(samples, is_freq_domain, name=os.path.splitext(os.path.basename(filepath))[0])

    def indices(self):
        if self.values.size == 0:
            return np.array([], dtype=int)
        return np.arange(self.start, self.start + self.values.size)

    def copy(self, name=None):
        new_name = name if name else self.name
        return Signal(self.values.copy(), self.start, new_name)

    def scaled(self, factor, name=None):
        new_name = name if name else f"{self.name}_x{factor}"
        return Signal(self.values * factor, self.start, new_name)
    
    def shifted(self, k, name=None):
        # x(n - k) => advance by k (start increases by k), x(n + k) => delay by k (start decreases)
        # We'll implement shifting semantics so that calling shifted(k) returns x(n - k)
        # where k positive means advance (the samples move left in n).
        new_start = self.start - k
        new_name = name if name else f"{self.name}_shift_{k}"
        return Signal(self.values.copy(), new_start, new_name)

    def folded(self, name=None):
        # x(-n): reverse samples and change start index
        new_name = name if name else f"{self.name}_fold"
        if self.values.size == 0:
            return Signal(np.array([]), 0, new_name)
        # original indices: start ... start+L-1
        L = self.values.size
        new_values = self.values[::-1].copy()
        # new start = - (old_end)
        old_end = self.start + L - 1
        new_start = -old_end
        return Signal(new_values, new_start, new_name)

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
            return Signal(np.array([]), 0, name)

        # Compute global start and end
        starts = [s.start for s in all_signals if s.values.size > 0]
        ends = [s.start + s.values.size - 1 for s in all_signals if s.values.size > 0]
        if not starts:
            return Signal(np.array([]), 0, name)

        full_start = min(starts)
        full_end = max(ends)
        L = full_end - full_start + 1
        total = np.zeros(L, dtype=float)

        for s in all_signals:
            if s.values.size == 0:
                continue
            idx = s.start - full_start
            total[idx:idx + s.values.size] += s.values

        return Signal(total, full_start, name)

    def subtract(self, other: "Signal", name=None):
        """Subtract another signal from the current one and return a new aligned Signal."""
        if name is None:
            name = f"{self.name}_minus_{other.name}"

        a, b, full_start = self._align(self, other)
        return Signal(a - b, full_start, name)

    def quantize(self, levels: int = None, bits: int = None, name: str = None):
        """
        Uniform mid-rise quantization.

        Provide either `levels` (number of quantization levels) or `bits`.
        If `bits` is given, levels = 2**bits.

        Returns a tuple of Signals: (quantized_signal, error_signal, encoded_signal)
        - quantized_signal: values mapped to reconstruction levels
        - error_signal: original - quantized
        - encoded_signal: integer code (0..levels-1) as a signal
        """
        if levels is None and bits is None:
            raise ValueError("Either levels or bits must be provided for quantization.")
        if bits is not None:
            if bits < 1:
                raise ValueError("bits must be >= 1")
            levels = 2 ** bits
        if levels is None or levels < 2:
            raise ValueError("levels must be an integer >= 2")

        x = self.values
        if x.size == 0:
            # empty signals -> return empty signals with appropriate names
            q_name = name if name else f"{self.name}_quant_L{levels}"
            q = Signal(np.array([]), 0, q_name)
            e = Signal(np.array([]), 0, q_name + "_err")
            enc = Signal(np.array([]), 0, q_name + "_enc")
            return q, e, enc

        xmin = float(np.min(x))
        xmax = float(np.max(x))

        # If constant signal, quantized = original, error 0, encoded 0
        if xmax == xmin:
            q_vals = x.copy()
            err_vals = np.zeros_like(x)
            enc_vals = np.zeros_like(x, dtype=int)
        else:
            # step size (range divided by number of levels)
            Delta = (xmax - xmin) / float(levels)
            # map to level indices k in [0, levels-1]
            # k = floor((x - xmin) / Delta), but ensure xmax maps to levels-1
            k = np.floor((x - xmin) / Delta).astype(int)
            # handle edge where x == xmax -> floor gives levels, clamp to levels-1
            k = np.where(k >= levels, levels - 1, k)
            k = np.where(k < 0, 0, k)
            # mid-rise reconstruction level: xmin + (k + 0.5)*Delta
            q_vals = xmin + (k.astype(float) + 0.5) * Delta
            err_vals = x - q_vals
            enc_vals = k.astype(int)

        q_name = name if name else f"{self.name}_quant_L{levels}"
        q = Signal(q_vals, self.start, q_name)
        e = Signal(err_vals, self.start, q_name + "_err")
        enc = Signal(enc_vals.astype(float), self.start, q_name + "_enc")

        return q, e, enc

    def moving_avg(self, window_size: int, name: str = None):
        """Compute the moving average of the signal with the given window size."""
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.values.size == 0:
            new_name = name if name else f"{self.name}_movavg_{window_size}"
            return Signal(np.array([]), 0, new_name)

        cumsum = np.cumsum(np.insert(self.values, 0, 0)) 
        mov_avg_values = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

        # self.start + (window_size - 1) // 2
        new_name = name if name else f"{self.name}_movavg_{window_size}"
        return Signal(mov_avg_values, self.start, new_name)

    def derivative(self, name: str = None, order: int = 1):
        """Compute the discrete-time derivative of the signal. Order can be 1 or 2."""
        if self.values.size == 0:
            new_name = name if name else f"{self.name}_deriv{order}"
            return Signal(np.array([]), 0, new_name)

        if order == 1:
            # First Derivative: y(n) = x(n) - x(n-1)
            deriv_values = np.diff(self.values)
            # self.start + 1
            new_start = self.start
            new_name = name if name else f"{self.name}_deriv1"
        elif order == 2:
            # Second Derivative: y(n) = x(n+1) - 2*x(n) + x(n-1)
            if self.values.size < 3:
                new_name = name if name else f"{self.name}_deriv2"
                return Signal(np.array([]), 0, new_name)
            
            # Compute: x(n+1) - 2*x(n) + x(n-1)
            deriv_values = self.values[2:] - 2*self.values[1:-1] + self.values[:-2]
            # self.start + 1
            new_start = self.start
            new_name = name if name else f"{self.name}_deriv2"
        else:
            raise ValueError("order must be 1 or 2")
        
        return Signal(deriv_values, new_start, new_name)
    
    def convolve(self, other: "Signal", name: str = None, mode: Literal['same', 'full'] = 'full'):
        """Convolve the current signal with another signal.
            Check mode:
                'full' ->   return unchanged
                'same' ->   Slice result to length len(self.values)
                            Shift start index by (len(other.values) - 1) // 2
        """
        if self.values.size == 0 or other.values.size == 0:
            new_name = name if name else f"{self.name}_conv_{other.name}"
            return Signal(np.array([]), 0, new_name)
        
        # Array method
        _product = [[x * y for x in self.values] for y in other.values]
        n, m = len(self.values), len(other.values)
        conv_values = [0] * (n + m - 1) # Result length

        # Sum along diagonals
        for i in range(m):
            for j in range(n):
                conv_values[i + j] += _product[i][j]

        # conv_values = np.convolve(self.values, other.values, mode='full')
        full_result = np.asarray(conv_values, dtype=float)
        if mode == 'same':
            start_idx = other.values.size - 1
            full_result = full_result[start_idx:start_idx + self.values.size]
            new_start = self.start
        else:
            new_start = self.start + other.start
            
        new_name = name if name else f"{self.name}_conv_{other.name}"
        return Signal(full_result, new_start, new_name)

    def correlation(self, other: "Signal") -> "Signal":
        """Compute the cross-correlation of the current signal with another signal."""
        other_rev = Signal(other.values[::-1], -(other.start + other.values.size - 1), other.name, other.is_frequency_domain)
        raw_correlation = self.convolve(other_rev, name=f"{self.name}_corr_{other_rev.name}", mode='same')
        # actual = np.correlate(self.values, other.values, mode='same')
        
        # """Include only positive lags in the correlation result."""
        # raw_correlation.values = raw_correlation.values[raw_correlation.values.size // 2:]
        
        factor = math.sqrt(np.sum(self.values**2) * np.sum(other.values**2))
        normalized_correlation = np.divide(raw_correlation.values, factor)
        return Signal(normalized_correlation, raw_correlation.start, raw_correlation.name, raw_correlation.is_frequency_domain)

    @staticmethod
    def argmax_abs(arr):
        max_val = abs(arr[0])
        max_idx = 0

        for i in range(1, len(arr)):
            if abs(arr[i]) > max_val:
                max_val = abs(arr[i])
                max_idx = i

        return max_idx

    def __str__(self):
        return f"{self.name}: start={self.start}, len={self.values.size}"

    def fourier(self, idft: bool = False):
        """
        Compute DFT | IDFT of the signal using manual implementation.
        """
        new_name = f"{self.name}_{'idft' if idft else 'dft'}"
        
        if self.values.size == 0:
            return Signal(np.array([]), 0, new_name, not idft)
        
        N = self.values.size
        
        # Compute DFT/IDFT manually (smart implementation for both)
        sign = 1 if idft else -1
        
        X = np.zeros(N, dtype=complex)
        for k in range(N):
            for n in range(N):
                X[k] += self.values[n] * np.exp(sign * 2j * np.pi * k * n / N)
            X[k] = X[k] * (1 / N if idft else 1)
        
        if idft:
            X = np.real_if_close(X)
        
        result_signal = Signal(X, self.start, new_name, not idft)
        return result_signal
    
    def get_magnitude_and_phase_spectrum(self, sampling_freq: float):
        """
        Compute and return the magnitude spectrum (amplitude) of the signal.
        """
        if self.values.size == 0:
            return np.array([]), np.array([])

        X = self.values
        N = len(X)
        
        phases = np.array([math.atan2(X[i].imag, X[i].real) for i in range(N)])
        magnitudes = np.array([math.sqrt(X[i].real**2 + X[i].imag**2) for i in range(N)])
        
        # phases = np.angle()
        # magnitudes = np.abs()
        
        return magnitudes, phases