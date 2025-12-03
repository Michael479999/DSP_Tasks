# signal_tool.py
# Requires: PySide6, matplotlib, numpy
# pip install PySide6 matplotlib numpy

import os
import sys
from typing import List, Optional, Union
import uuid
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QListWidgetItem, QInputDialog, QLineEdit, QMenu
)
from base import get_file_path, save_signal
from my_signal import Signal
from matplot_canvas import MplCanvas
# from tests import (
#     AddSignalSamplesAreEqual,
#     SubSignalSamplesAreEqual,
#     MultiplySignalByConst,
#     ShiftSignalByConst,
#     Folding
# )

# -------------------------
# Main Application Window
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Discrete Signal Tool")
        self.resize(1000, 600)

        # data store
        self.signals: List[Signal] = []

        # central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Initialize Menu Bar
        self._create_menu_bar()

        # Left panel: controls and list
        left_panel = QtWidgets.QVBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        left_panel.addWidget(QtWidgets.QLabel("Loaded signals (select one or more):"))
        left_panel.addWidget(self.list_widget)

        # Buttons
        btn_layout = QtWidgets.QVBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_sub = QtWidgets.QPushButton("Subtract")
        self.btn_scale = QtWidgets.QPushButton("Multiply")
        self.btn_shift = QtWidgets.QPushButton("Shift")
        self.btn_fold = QtWidgets.QPushButton("Fold")
        self.btn_quant = QtWidgets.QPushButton("Quantize")
        self.btn_fourier = QtWidgets.QPushButton("Apply Fourier")
        self.btn_inverse_fourier = QtWidgets.QPushButton("Inverse Fourier")
        self.btn_moving_avg = QtWidgets.QPushButton("Moving Avg")
        self.btn_derivative = QtWidgets.QPushButton("Derivative")
        self.btn_convolve = QtWidgets.QPushButton("Convolve")
        self.btn_delete = QtWidgets.QPushButton("Delete")

        # ===== Basic Operations Group =====
        basic_group = QtWidgets.QGroupBox("Basic Operations")
        basic_layout = QtWidgets.QGridLayout()
        basic_layout.addWidget(self.btn_add, 0, 0)
        basic_layout.addWidget(self.btn_sub, 0, 1)
        basic_layout.addWidget(self.btn_scale, 1, 0)
        basic_layout.addWidget(self.btn_delete, 1, 1)
        basic_group.setLayout(basic_layout)

        # ===== Transform Operations Group =====
        transform_group = QtWidgets.QGroupBox("Transform Operations")
        transform_layout = QtWidgets.QGridLayout()
        transform_layout.addWidget(self.btn_shift, 0, 0)
        transform_layout.addWidget(self.btn_fold, 0, 1)
        transform_layout.addWidget(self.btn_quant, 1, 0)
        transform_group.setLayout(transform_layout)
        
        # ===== Fourier Operations Group =====
        fourier_group = QtWidgets.QGroupBox("Fourier Operations")
        fourier_layout = QtWidgets.QGridLayout()
        fourier_layout.addWidget(self.btn_fourier, 0, 0)
        fourier_layout.addWidget(self.btn_inverse_fourier, 0, 1)
        fourier_group.setLayout(fourier_layout)

        # ===== Advanced Operations Group =====
        advanced_group = QtWidgets.QGroupBox("Advanced Operations")
        advanced_layout = QtWidgets.QGridLayout()
        advanced_layout.addWidget(self.btn_moving_avg, 0, 0)
        advanced_layout.addWidget(self.btn_derivative, 0, 1)
        advanced_layout.addWidget(self.btn_convolve, 1, 0)
        advanced_group.setLayout(advanced_layout)

        # ---- Add all groups to main btn_layout ----
        btn_layout.addWidget(basic_group)
        btn_layout.addWidget(transform_group)
        btn_layout.addWidget(fourier_group)
        btn_layout.addWidget(advanced_group)

        # Add to left panel
        left_panel.addLayout(btn_layout)

        # Save/export buttons (future extension)
        left_panel.addStretch()
        main_layout.addLayout(left_panel, 0)

        # Right panel: plot area
        right_panel = QtWidgets.QVBoxLayout()
        self.canvas = MplCanvas(self)
        right_panel.addWidget(self.canvas)
        main_layout.addLayout(right_panel, 1)

        # Connect signals
        self.btn_add.clicked.connect(self.add_selected)
        self.btn_sub.clicked.connect(self.subtract_selected)
        self.btn_scale.clicked.connect(self.scale_selected)
        self.btn_shift.clicked.connect(self.shift_selected)
        self.btn_fold.clicked.connect(self.fold_selected)
        self.btn_quant.clicked.connect(self.quantize_selected)
        self.btn_fourier.clicked.connect(lambda: self.apply_fourier())
        self.btn_inverse_fourier.clicked.connect(lambda: self.apply_fourier(inverse=True))
        self.btn_moving_avg.clicked.connect(self.moving_avg_selected)
        self.btn_derivative.clicked.connect(self.derivative_selected)
        self.btn_convolve.clicked.connect(self.convolve_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.list_widget.itemDoubleClicked.connect(self.rename_item)


    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        
        # --- Load Menu ---
        load_action = menu_bar.addAction("Load")
        load_action.triggered.connect(self.load_signals)
        
        # --- Create Menu ---
        create_menu = menu_bar.addMenu("Create")

        # Signal Generation submenu
        sine_action = QtGui.QAction("Sine Wave", self)
        cosine_action = QtGui.QAction("Cosine Wave", self)

        create_menu.addAction(sine_action)
        create_menu.addAction(cosine_action)

        sine_action.triggered.connect(lambda: self._generate_signal("sine"))
        cosine_action.triggered.connect(lambda: self._generate_signal("cosine"))
        
        # --- Mode Menu ---
        mode_menu = menu_bar.addMenu("Mode")

        self.continuous_action = QtGui.QAction("Continuous", self)
        self.discrete_action = QtGui.QAction("Discrete", self)

        self.continuous_action.setCheckable(True)
        self.discrete_action.setCheckable(True)

        mode_group = QtGui.QActionGroup(self)
        mode_group.addAction(self.continuous_action)
        mode_group.addAction(self.discrete_action)
        mode_group.setExclusive(True)

        self.continuous_action.setChecked(True)

        mode_menu.addAction(self.continuous_action)
        mode_menu.addAction(self.discrete_action)

        # --- Plot Menu ---
        plot_menu = menu_bar.addMenu("Plot")
        
        self.plot_magnitude_phase_spectrum_action = QtGui.QAction("Magnitude & Phase Spectrum", self)
        self.plot_selected_action = QtGui.QAction("Selected", self)
        self.plot_all_action = QtGui.QAction("All", self)

        plot_menu.addAction(self.plot_magnitude_phase_spectrum_action)
        plot_menu.addAction(self.plot_selected_action)
        plot_menu.addAction(self.plot_all_action)

        self.plot_magnitude_phase_spectrum_action.triggered.connect(lambda: self.display_magnitude_phase_spectrum())
        self.plot_selected_action.triggered.connect(lambda:self.plot_selected(self.signals))
        self.plot_all_action.triggered.connect(lambda: self.plot_all(self.signals))

    # -------------------------
    # Signal generation
    # -------------------------
    def _generate_signal(self, signal_type: str):
        """Prompt user for parameters and generate sine/cosine signal."""
        try:
            A, ok = QInputDialog.getDouble(self, "Amplitude", "Enter amplitude (A):", 1.0, 0.0)
            if not ok: return

            D, ok = QInputDialog.getDouble(self, "Y-Offset", "Enter offset (D):", 0.0)
            if not ok: return

            theta, ok = QInputDialog.getDouble(self, "Phase Shift", "Enter phase shift (θ in radians):", 0.0)
            if not ok: return

            f, ok = QInputDialog.getDouble(self, "Analog Frequency", "Enter analog frequency (Hz):", 5.0, 0.1)
            if not ok: return

            fs, ok = QInputDialog.getDouble(self, "Sampling Frequency", "Enter sampling frequency (Hz):", 50.0, 0.1)
            if not ok: return

            # --- Nyquist Theorem Validation ---
            if fs < 2 * f:
                QMessageBox.warning(
                    self,
                    "Sampling Error",
                    f"Sampling frequency must be at least twice the maximum frequency (fs >= 2f).\n"
                    f"Current: fs = {fs}, 2f = {2*f}",
                )
                return

            # --- Generate signal ---
            t = np.arange(0, 1, 1/fs)  # 1-second duration
            if signal_type == "sine":
                y = A * np.sin(2 * np.pi * f * t + theta) + D
            else:
                y = A * np.cos(2 * np.pi * f * t + theta) + D

            file_name = f"{signal_type}_{A}A_{f}Hz_{D}D_{uuid.uuid4()}"
            created_signal = Signal(y, D, file_name)

            self.add_signal_to_list(created_signal)
            self.refresh_list_items()

            confirm = QMessageBox.information(
                self,
                "Signal Created",
                (
                    f"{file_name}\nSamples: {len(y)}\n\n"
                    "Would you like to plot this signal?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            save_signal(created_signal.name, created_signal)

            if confirm == QMessageBox.StandardButton.No:
                return
            
            self.plot_all([created_signal])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # -------------------------
    # Utilities for list
    # -------------------------
    def add_signal_to_list(self, sig: Signal):
        self.signals.append(sig)
        item = QListWidgetItem(f"{sig.name}  (start={sig.start if not sig.is_frequency_domain else 0}, len={sig.values.size})")
        item.setData(QtCore.Qt.ItemDataRole.UserRole, len(self.signals) - 1)  # index in self.signals
        self.list_widget.addItem(item)
    
    def add_signals_to_list(self, sigs: List[Signal]):
        [self.add_signal_to_list(sig) for sig in sigs]

    def refresh_list_items(self):
        self.list_widget.clear()
        for idx, s in enumerate(self.signals):
            item = QListWidgetItem(f"{s.name}  (start={s.start}, len={s.values.size})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, idx)
            self.list_widget.addItem(item)

    def get_selected_indices(self) -> List[int]:
        items = self.list_widget.selectedItems()
        indices = []
        for it in items:
            idx = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(idx, int):
                indices.append(idx)
        return indices

    # -------------------------
    # Button callbacks
    # -------------------------
    def handle_signal_load(self, result: Union[Signal, Exception]):
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Error reading file", str(result))
            return

        new_name, ok = QInputDialog.getText(
            self,
            "Signal name",
            "Enter name for signal:",
            QLineEdit.EchoMode.Normal,
            result.name
        )
        
        if ok and new_name.strip():
            result.name = new_name.strip()

    def load_signals(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Open signal file(s)", "", "Text files (*.txt);;All files (*)")
        if not fnames:
            return
        
        self.add_signals_to_list(Signal.from_files(fnames, self.handle_signal_load))

    def plot_selected(self, signals: List[Signal]):
        idxs = self.get_selected_indices()
        if not idxs:
            QMessageBox.information(self, "No selection", "Select one or more signals to plot.")
            return
        s = [signals[i] for i in idxs]
        if len(s) == 1:
            self.canvas.plot_signal(s[0], self.discrete_action.isChecked())
        else:
            self.canvas.plot_multiple(s, self.discrete_action.isChecked())

    def plot_all(self, signals: List[Signal]):
        if not signals:
            QMessageBox.information(self, "No signals", "No loaded signals.")
            return
        self.canvas.plot_multiple(signals, self.discrete_action.isChecked())

    def add_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) < 1:
            QMessageBox.information(self, "Select signals", "Select at least one signal to add.")
            return
        signals = [self.signals[i].copy() for i in idxs]
        res = signals[0].add(signals[1:])
        self.add_signal_to_list(res)
        self.refresh_list_items()
        
        save_signal(res.name, res)
        QMessageBox.information(self, "Saved", f"Addition result saved to current script directory as: {res.name}.txt")

    def scale_selected(self):
        idxs = self.get_selected_indices()
        if not idxs:
            QMessageBox.information(self, "Select", "Select a single signal to scale.")
            return
        if len(idxs) > 1:
            QMessageBox.information(self, "Select one", "Select exactly one signal to scale.")
            return
        sig = self.signals[idxs[0]]
        factor_s, ok = QInputDialog.getText(self, "Scale factor", "Enter scale factor (e.g., 5):", QLineEdit.EchoMode.Normal, "1.0")
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
        
        save_signal(res.name, res)
        QMessageBox.information(self, "Saved", f"Multiply result saved to current script directory as: {res.name}.txt")

    def subtract_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) != 2:
            QMessageBox.information(self, "Select two", "Select exactly two signals to subtract (A - B).")
            return
        a: Signal = self.signals[idxs[0]]
        b: Signal = self.signals[idxs[1]]
        res = a.subtract(b)
        self.add_signal_to_list(res)
        self.refresh_list_items()

        save_signal(res.name, res)
        QMessageBox.information(self, "Saved", f"Subtraction result saved to current script directory as: {res.name}.txt")

    def shift_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) != 1:
            QMessageBox.information(self, "Select one", "Select a single signal to shift.")
            return
        sig: Signal = self.signals[idxs[0]]
        k_s, ok = QInputDialog.getText(self, "Shift k", "Enter integer k:", QLineEdit.EchoMode.Normal, "1")
        if not ok:
            return
        try:
            k = int(k_s)
            assert(k != 0)
        except:
            QMessageBox.critical(self, "Bad k-step shift value", "k must be integer not equal to 0.")
            return
        
        filename = f"{sig.name + "_advance_" + str(abs(k)) if k > 0 else sig.name + "_delay_" + str(abs(k))}"

        res = sig.shifted(k, name=filename)
        self.add_signal_to_list(res)
        self.refresh_list_items()
        
        save_signal(filename, res)
        QMessageBox.information(self, "Saved", f"Shift result saved to current script directory as: {filename}.txt")

    def fold_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) != 1:
            QMessageBox.information(self, "Select one", "Select one signal to fold.")
            return
        sig = self.signals[idxs[0]]
        res = sig.folded(name=f"{sig.name}_fold")
        self.add_signal_to_list(res)
        self.refresh_list_items()
        
        save_signal(res.name, res)
        QMessageBox.information(self, "Saved", f"Folding result saved to current script directory as: {res.name}.txt")

    def quantize_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) != 1:
            QMessageBox.information(self, "Select one", "Select a single signal to quantize.")
            return
        sig: Signal = self.signals[idxs[0]]

        # Ask user whether to enter bits or levels
        choice, ok = QInputDialog.getItem(self, "Quantize Mode", "Choose input type:", ["bits", "levels"], 0, False)
        if not ok:
            return

        if choice == "bits":
            b_s, ok = QInputDialog.getInt(self, "Bits", "Enter number of bits:", 8, 1)
            if not ok:
                return
            bits = int(b_s)
            levels = None
        else:
            levels, ok = QInputDialog.getInt(self, "Levels", "Enter number of quantization levels:", 256, 2)
            if not ok:
                return
            bits = None

        try:
            q, e, enc = sig.quantize(levels=levels, bits=bits, name=f"{sig.name}_quantized")
        except Exception as ex:
            QMessageBox.critical(self, "Quantization error", str(ex))
            return

        # Add to list and save files
        self.add_signal_to_list(q)
        self.add_signal_to_list(e)
        self.add_signal_to_list(enc)
        self.refresh_list_items()

        save_signal(q.name, q)
        save_signal(e.name, e)
        save_signal(enc.name, enc)

        QMessageBox.information(self, "Saved", f"Quantized, error and encoded signals saved to current script directory as: {q.name}.txt, {e.name}.txt, {enc.name}.txt")

        # Plot: quantized and error and encoded
        self.canvas.plot_multiple([q, e, enc], self.discrete_action.isChecked())

    def apply_fourier(self, inverse: bool = False):
        idxs = self.get_selected_indices()
        if len(idxs) != 1:
            QMessageBox.information(self, "Select one", "Select a single signal to transform.")
            return
        
        sig: Signal = self.signals[idxs[0]]
        res = sig.fourier(inverse=inverse)
        
        self.add_signal_to_list(res)
        self.refresh_list_items()
        save_signal(res.name, res)
        QMessageBox.information(self, "Saved", f"Fourier result saved to current script directory as: {res.name}.txt\n", QMessageBox.StandardButton.Ok)
        self.canvas.plot_multiple([sig, res], self.discrete_action.isChecked())
        
        if inverse:
            return
        
        reply = QMessageBox.question(self, "Plotting Note", "Frequency and Phase plots are now available, Would you like to view them?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.display_magnitude_phase_spectrum(res)

    def display_magnitude_phase_spectrum(self, signal: Optional[Signal] = None):
        if signal is None:
            idxs = self.get_selected_indices()
            if len(idxs) != 1:
                QMessageBox.information(self, "Select one", "Select a single signal to plot its magnitude and phase spectrum.")
                return
            signal = self.signals[idxs[0]]
            
        
        if not signal.is_frequency_domain:
            QMessageBox.information(self, "Not in frequency domain", "Selected signal is not in frequency domain. Please apply Fourier Transform first.")
            return
        
        sf, ok = QInputDialog.getDouble(self, "Sampling Frequency", "Enter sampling frequency (Hz):", 50.0, 0.1)
        if not ok:
            return
        
        if sf <= 0:
            raise ValueError("Sampling frequency must be positive")
        
        MplCanvas.plot_frequency_domain(signal, sf)

    def moving_avg_selected(self):
        try:
            idxs = self.get_selected_indices()
            assert len(idxs) == 1, "Select exactly one signal for moving average."
            idx = idxs[0]
            curr = self.signals[idx]
            k_s, ok = QInputDialog.getInt(self, "Window size", "Enter window size:", 3, 1)
            if not ok:
                return
            
            k = int(k_s)
            assert k > 0, "Window size must be positive."
            res = curr.moving_avg(k)
            
            self.add_signal_to_list(res)
            self.refresh_list_items()
            save_signal(res.name, res)
            
            QMessageBox.information(self, "Saved", f"Moving average result saved to current script directory as: {res.name}.txt")
            
            self.canvas.plot_multiple([curr, res], self.discrete_action.isChecked())
        except AssertionError as e:
            QMessageBox.information(self, "Warning", str(e))
    
    def derivative_selected(self):
        try:
            idxs = self.get_selected_indices()
            assert len(idxs) == 1, "Select exactly one signal for derivative."
            idx = idxs[0]
            curr = self.signals[idx]
            
            order, ok = QInputDialog.getInt(self, "Derivative Order", "Enter order (1 or 2):", 1, 1, 2)
            if not ok:
                return
            
            res = curr.derivative(order=order)
            
            self.add_signal_to_list(res)
            self.refresh_list_items()
            save_signal(res.name, res)
            
            QMessageBox.information(self, "Saved", f"Derivative result saved to current script directory as: {res.name}.txt")
            
            self.canvas.plot_multiple([curr, res], self.discrete_action.isChecked())
        except AssertionError as e:
            QMessageBox.information(self, "Warning", str(e))
    def convolve_selected(self):
        try:
            idxs = self.get_selected_indices()
            assert len(idxs) == 2, "Select exactly two signals for convolution."
            idx1, idx2 = idxs[0], idxs[1]
            sig1, sig2 = self.signals[idx1], self.signals[idx2]
            
            res = sig1.convolve(sig2)
            
            self.add_signal_to_list(res)
            self.refresh_list_items()
            save_signal(res.name, res)
            
            QMessageBox.information(self, "Saved", f"Convolution result saved to current script directory as: {res.name}.txt")
            
            self.canvas.plot_multiple([sig1, sig2, res], self.discrete_action.isChecked())
        except AssertionError as e:
            QMessageBox.information(self, "Warning", str(e))

    def delete_selected(self):
        idxs = sorted(self.get_selected_indices(), reverse=True)
        if not idxs:
            QMessageBox.information(self, "No Selection", "Please select one or more signals to delete.")
            return

        confirm = QMessageBox.question(
            self,
            "Delete Signals",
            (
                f"Do you also want to delete the selected file(s) from your computer?\n\n"
                "- Click 'Yes' to delete from file system.\n"
                "- Click 'No' to remove only from the memory."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )

        if confirm == QMessageBox.StandardButton.Cancel:
            return

        delete_from_disk = confirm == QMessageBox.StandardButton.Yes
        deleted_count, failed = 0, []

        for i in idxs:
            sig = self.signals[i]
            file_path = get_file_path(sig.name)
            
            try:
                if delete_from_disk and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        failed.append((file_path, str(e)))
                del self.signals[i]
                deleted_count += 1
            except Exception as e:
                failed.append((file_path, str(e)))

        self.refresh_list_items()

        if failed:
            msg = f"{deleted_count} signal(s) removed.\n\nSome files could not be deleted:\n"
            msg += "\n".join([f"• {name}: {err}" for name, err in failed])
            QMessageBox.warning(self, "Partial Deletion", msg)
            return
        msg_type = "and deleted from disk" if delete_from_disk else "from the list only"
        QMessageBox.information(
            self,
            "Deletion Complete",
            f"{deleted_count} signal(s) successfully removed {msg_type}."
        )

    def rename_item(self, item):
        idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if idx is None: return
        
        sig = self.signals[idx]
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", QLineEdit.EchoMode.Normal, sig.name)
        if not ok or not new_name.strip(): return
        
        sig.name = new_name.strip()
        self.refresh_list_items()

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
