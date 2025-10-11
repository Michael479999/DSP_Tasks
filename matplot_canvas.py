from typing import List
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from my_signal import Signal

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(figsize=(6, 4), dpi=dpi, tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_signal(self, signal: 'Signal', is_discrete: bool):
        """Plot a single signal, either as discrete (stem) or continuous (line)."""
        self.ax.clear()

        if signal.values.size == 0:
            self.ax.set_title(f"{signal.name} (empty)")
            self.draw()
            return

        n = signal.indices()
        self.ax.set_title(signal.name)
        self.ax.set_xlabel("n")
        self.ax.set_ylabel("Amplitude")

        if is_discrete:
            markerline, stemlines, baseline = self.ax.stem(n, signal.values)
            plt.setp(markerline, color='C0')
            plt.setp(stemlines, color='C0')
        else:
            self.ax.plot(n, signal.values, color='C0', linewidth=1.8)

        self.ax.grid(True)
        self.draw()

    def plot_multiple(self, signals: List['Signal'], is_discrete: bool):
        """Plot multiple signals in discrete or continuous mode."""
        self.ax.clear()

        if not signals:
            self.ax.set_title("No signals to plot")
            self.draw()
            return

        starts = [s.start for s in signals if s.values.size > 0]
        ends = [s.start + s.values.size - 1 for s in signals if s.values.size > 0]

        if not starts:
            self.ax.set_title("All signals empty")
            self.draw()
            return

        full_start = min(starts)
        full_end = max(ends)
        n = np.arange(full_start, full_end + 1)

        colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))

        for s, color in zip(signals, colors):
            if s.values.size == 0:
                continue

            arr = np.zeros(n.size)
            idx = s.start - full_start
            arr[idx:idx + s.values.size] = s.values

            if is_discrete:
                markerline, stemlines, baseline = self.ax.stem(n, arr, label=s.name)
                plt.setp(markerline, color=color)
                plt.setp(stemlines, color=color)
            else:
                self.ax.plot(n, arr, label=s.name, color=color, linewidth=1.8)

        self.ax.set_xlabel("n")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend()
        self.ax.grid(True)
        self.draw()