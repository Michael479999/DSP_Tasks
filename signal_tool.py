# signal_tool.py
# Requires: PySide6, matplotlib, numpy
# pip install PySide6 matplotlib numpy

import sys
import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QListWidgetItem, QInputDialog, QLineEdit
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from tests import (
    AddSignalSamplesAreEqual,
    SubSignalSamplesAreEqual,
    MultiplySignalByConst,
    ShiftSignalByConst,
    Folding
)

# -------------------------
# Signal representation
# -------------------------
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
        new_start = self.start + k
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

# -------------------------
# Matplotlib canvas for Qt
# -------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(figsize=(6, 4), dpi=dpi, tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_signal(self, signal: Signal):
        self.ax.clear()
        if signal.values.size == 0:
            self.ax.set_title(f"{signal.name} (empty)")
            self.draw()
            return
        n = signal.indices()
        # stem plot
        markerline, stemlines, baseline = self.ax.stem(n, signal.values)
        self.ax.set_xlabel("n")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title(signal.name)
        self.ax.grid(True)
        self.draw()

    def plot_multiple(self, signals):
        self.ax.clear()
        if not signals:
            self.ax.set_title("No signals to plot")
            self.draw()
            return
        # compute global start/end
        starts = [s.start for s in signals if s.values.size > 0]
        ends = [s.start + s.values.size - 1 for s in signals if s.values.size > 0]
        if not starts:
            self.ax.set_title("All signals empty")
            self.draw()
            return
        full_start = min(starts)
        full_end = max(ends)
        n = np.arange(full_start, full_end + 1)
        for s in signals:
            arr = np.zeros(n.size)
            if s.values.size > 0:
                idx = s.start - full_start
                arr[idx:idx + s.values.size] = s.values
            self.ax.stem(n, arr, label=s.name, use_line_collection=True)
        self.ax.set_xlabel("n")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend()
        self.ax.grid(True)
        self.draw()

# -------------------------
# Main Application Window
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Discrete Signal Tool")
        self.resize(1000, 600)

        # central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Left panel: controls and list
        left_panel = QtWidgets.QVBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        left_panel.addWidget(QtWidgets.QLabel("Loaded signals (select one or more):"))
        left_panel.addWidget(self.list_widget)

        # Buttons
        btn_layout = QtWidgets.QGridLayout()
        self.btn_load = QtWidgets.QPushButton("Load signal from file")
        self.btn_plot = QtWidgets.QPushButton("Plot selected")
        self.btn_plot_all = QtWidgets.QPushButton("Plot all")
        self.btn_add = QtWidgets.QPushButton("Add selected -> new")
        self.btn_scale = QtWidgets.QPushButton("Multiply selected by const")
        self.btn_sub = QtWidgets.QPushButton("Subtract (A - B)")
        self.btn_shift = QtWidgets.QPushButton("Delay/Advance selected")
        self.btn_fold = QtWidgets.QPushButton("Fold selected")
        self.btn_delete = QtWidgets.QPushButton("Delete selected")
        btn_layout.addWidget(self.btn_load, 0, 0)
        btn_layout.addWidget(self.btn_plot, 0, 1)
        btn_layout.addWidget(self.btn_plot_all, 1, 0)
        btn_layout.addWidget(self.btn_add, 1, 1)
        btn_layout.addWidget(self.btn_scale, 2, 0)
        btn_layout.addWidget(self.btn_sub, 2, 1)
        btn_layout.addWidget(self.btn_shift, 3, 0)
        btn_layout.addWidget(self.btn_fold, 3, 1)
        btn_layout.addWidget(self.btn_delete, 4, 0)
        left_panel.addLayout(btn_layout)

        # Save/export buttons (future extension)
        left_panel.addStretch()
        main_layout.addLayout(left_panel, 0)

        # Right panel: plot area
        right_panel = QtWidgets.QVBoxLayout()
        self.canvas = MplCanvas(self, dpi=100)
        right_panel.addWidget(self.canvas)
        main_layout.addLayout(right_panel, 1)

        # data store
        self.signals = []  # list of Signal objects

        # Connect signals
        self.btn_load.clicked.connect(self.load_signal)
        self.btn_plot.clicked.connect(self.plot_selected)
        self.btn_plot_all.clicked.connect(self.plot_all)
        self.btn_add.clicked.connect(self.add_selected)
        self.btn_scale.clicked.connect(self.scale_selected)
        self.btn_sub.clicked.connect(self.subtract_selected)
        self.btn_shift.clicked.connect(self.shift_selected)
        self.btn_fold.clicked.connect(self.fold_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.list_widget.itemDoubleClicked.connect(self.rename_item)

    # -------------------------
    # Utilities for list
    # -------------------------
    def add_signal_to_list(self, sig: Signal):
        self.signals.append(sig)
        item = QListWidgetItem(f"{sig.name}  (start={sig.start}, len={sig.values.size})")
        item.setData(QtCore.Qt.UserRole, len(self.signals) - 1)  # index in self.signals
        self.list_widget.addItem(item)

    def refresh_list_items(self):
        self.list_widget.clear()
        for idx, s in enumerate(self.signals):
            item = QListWidgetItem(f"{s.name}  (start={s.start}, len={s.values.size})")
            item.setData(QtCore.Qt.UserRole, idx)
            self.list_widget.addItem(item)

    def get_selected_indices(self):
        items = self.list_widget.selectedItems()
        indices = []
        for it in items:
            idx = it.data(QtCore.Qt.UserRole)
            if isinstance(idx, int):
                indices.append(idx)
        return indices

    # -------------------------
    # Button callbacks
    # -------------------------
    def load_signal(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open signal file", "", "Text files (*.txt);;All files (*)")
        if not fname:
            return
        try:
            sig = Signal.from_file(fname)
        except Exception as e:
            QMessageBox.critical(self, "Error reading file", str(e))
            return
        # allow renaming
        new_name, ok = QInputDialog.getText(self, "Signal name", "Enter name for signal:", QLineEdit.Normal, sig.name)
        if ok and new_name.strip():
            sig.name = new_name.strip()
        self.add_signal_to_list(sig)

    def plot_selected(self):
        idxs = self.get_selected_indices()
        if not idxs:
            QMessageBox.information(self, "No selection", "Select one or more signals to plot.")
            return
        s = [self.signals[i] for i in idxs]
        if len(s) == 1:
            self.canvas.plot_signal(s[0])
        else:
            self.canvas.plot_multiple(s)

    def plot_all(self):
        if not self.signals:
            QMessageBox.information(self, "No signals", "No loaded signals.")
            return
        self.canvas.plot_multiple(self.signals)

    def add_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) < 1:
            QMessageBox.information(self, "Select signals", "Select at least one signal to add.")
            return
        signals = [self.signals[i].copy() for i in idxs]
        result = Signal.add(signals, name="sum")
        self.add_signal_to_list(result)
        self.refresh_list_items()
        # auto-save
        save_signal_to_file("add.txt", result)

        AddSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", result.indices(), result.values)

        QMessageBox.information(self, "Saved", "Addition result saved to add.txt")

    def scale_selected(self):
        idxs = self.get_selected_indices()
        if not idxs:
            QMessageBox.information(self, "Select", "Select a single signal to scale.")
            return
        if len(idxs) > 1:
            QMessageBox.information(self, "Select one", "Select exactly one signal to scale.")
            return
        sig = self.signals[idxs[0]]
        factor_s, ok = QInputDialog.getText(self, "Scale factor", "Enter scale factor (e.g., 5):", QLineEdit.Normal, "1.0")
        if not ok:
            return
        try:
            factor = float(factor_s)
        except:
            QMessageBox.critical(self, "Bad value", "Scale factor must be a number.")
            return
        res = sig.scaled(factor, name=f"{sig.name}_x{factor}")
        self.add_signal_to_list(res)
        self.refresh_list_items()
        # auto-save only if factor == 5 (test file requirement)
        if abs(factor - 5) < 1e-9:
            save_signal_to_file("mul5.txt", res)

            MultiplySignalByConst(5, res.indices(), res.values)
            
            QMessageBox.information(self, "Saved", "Multiply result saved to mul5.txt")

    def subtract_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) != 2:
            QMessageBox.information(self, "Select two", "Select exactly two signals to subtract (A - B).")
            return
        a = self.signals[idxs[0]]
        b = self.signals[idxs[1]]
        res = Signal.subtract(a, b, name=f"{a.name}_minus_{b.name}")
        self.add_signal_to_list(res)
        self.refresh_list_items()
        # auto-save
        save_signal_to_file("subtract.txt", res)

        SubSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", res.indices(), res.values)

        QMessageBox.information(self, "Saved", "Subtraction result saved to subtract.txt")

    def shift_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) != 1:
            QMessageBox.information(self, "Select one", "Select a single signal to shift.")
            return
        sig = self.signals[idxs[0]]
        k_s, ok = QInputDialog.getText(self, "Shift k", "Enter integer k:", QLineEdit.Normal, "1")
        if not ok:
            return
        try:
            k = int(k_s)
        except:
            QMessageBox.critical(self, "Bad value", "k must be integer.")
            return
        res = sig.shifted(k, name=f"{sig.name}_shift_{k}")
        self.add_signal_to_list(res)
        self.refresh_list_items()
        # auto-save to delay/advance files
        if k == 3:
            save_signal_to_file("advance3.txt", res)
            ShiftSignalByConst(3, res.indices(), res.values)
            QMessageBox.information(self, "Saved", "Shift result saved to advance3.txt")
        elif k == -3:
            save_signal_to_file("delay3.txt", res)
            ShiftSignalByConst(-3, res.indices(), res.values)
            QMessageBox.information(self, "Saved", "Shift result saved to delay3.txt")

    def fold_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) != 1:
            QMessageBox.information(self, "Select one", "Select one signal to fold.")
            return
        sig = self.signals[idxs[0]]
        res = sig.folded(name=f"{sig.name}_fold")
        self.add_signal_to_list(res)
        self.refresh_list_items()
        # auto-save
        save_signal_to_file("folding.txt", res)

        Folding(res.indices(), res.values)

        QMessageBox.information(self, "Saved", "Folding result saved to folding.txt")

    def delete_selected(self):
        idxs = sorted(self.get_selected_indices(), reverse=True)
        if not idxs:
            QMessageBox.information(self, "Select", "Select signals to delete.")
            return
        for i in idxs:
            try:
                del self.signals[i]
            except IndexError:
                pass
        self.refresh_list_items()

    def rename_item(self, item):
        idx = item.data(QtCore.Qt.UserRole)
        if idx is None:
            return
        sig = self.signals[idx]
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", QLineEdit.Normal, sig.name)
        if ok and new_name.strip():
            sig.name = new_name.strip()
            self.refresh_list_items()


def save_signal_to_file(filename, sig: Signal):
    with open(filename, "w") as f:
     f.write(f"{len(sig.values)}\n")
     for i, v in zip(sig.indices(), sig.values):
         f.write(f"{i} {v}\n")
# -------------------------
# Run app
# -------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
