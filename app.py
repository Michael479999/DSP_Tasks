# signal_tool.py
# Requires: PySide6, matplotlib, numpy
# pip install PySide6 matplotlib numpy

import os
import sys
from typing import List, Union
import uuid
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QListWidgetItem, QInputDialog, QLineEdit, QMenu, QSystemTrayIcon
)
from my_signal import Signal
from matplot_canvas import MplCanvas
from tests import (
    AddSignalSamplesAreEqual,
    SubSignalSamplesAreEqual,
    MultiplySignalByConst,
    ShiftSignalByConst,
    Folding
)

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
        btn_layout = QtWidgets.QGridLayout()
        self.btn_load = QtWidgets.QPushButton("Load signal from file")
        self.btn_add = QtWidgets.QPushButton("Add selected -> new")
        self.btn_scale = QtWidgets.QPushButton("Multiply selected by const")
        self.btn_sub = QtWidgets.QPushButton("Subtract (A - B)")
        self.btn_shift = QtWidgets.QPushButton("Delay/Advance selected")
        self.btn_fold = QtWidgets.QPushButton("Fold selected")
        self.btn_delete = QtWidgets.QPushButton("Delete selected")
        btn_layout.addWidget(self.btn_load, 0, 0)
        btn_layout.addWidget(self.btn_add, 0, 1)
        btn_layout.addWidget(self.btn_scale, 1, 0)
        btn_layout.addWidget(self.btn_sub, 1, 1)
        btn_layout.addWidget(self.btn_shift, 2, 0)
        btn_layout.addWidget(self.btn_fold, 2, 1)
        btn_layout.addWidget(self.btn_delete, 3, 0)
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
        self.btn_load.clicked.connect(self.load_signals)
        self.btn_add.clicked.connect(self.add_selected)
        self.btn_scale.clicked.connect(self.scale_selected)
        self.btn_sub.clicked.connect(self.subtract_selected)
        self.btn_shift.clicked.connect(self.shift_selected)
        self.btn_fold.clicked.connect(self.fold_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.list_widget.itemDoubleClicked.connect(self.rename_item)


    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        
        # --- Create Menu ---
        create_menu = menu_bar.addMenu("Create")

        # Signal Generation submenu
        signal_menu = QMenu("Signal Generation", self)
        sine_action = QtGui.QAction("Sine Wave", self)
        cosine_action = QtGui.QAction("Cosine Wave", self)

        signal_menu.addAction(sine_action)
        signal_menu.addAction(cosine_action)
        create_menu.addMenu(signal_menu)

        sine_action.triggered.connect(lambda: self._generate_signal("sine"))
        cosine_action.triggered.connect(lambda: self._generate_signal("cosine"))
        
        # --- Tools Menu ---
        tools_menu = menu_bar.addMenu("Tools")

        # Mode Menu
        mode_menu = QMenu("Mode", self)

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

        tools_menu.addMenu(mode_menu)

        # Plot Menu
        plot_menu = QMenu("Plot", self)
        
        self.plot_selected_action = QtGui.QAction("Selected", self)
        self.plot_all_action = QtGui.QAction("All", self)

        plot_menu.addAction(self.plot_selected_action)
        plot_menu.addAction(self.plot_all_action)
        
        tools_menu.addMenu(plot_menu)

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

            file_name = f"{signal_type}_{A}A_{f}Hz_{D}D_{uuid.uuid4()}.txt"
            created_signal = Signal(y, D, os.path.join(os.getcwd(), file_name), file_name)

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

            save_signal_to_file(f"{created_signal.name}.txt", created_signal)

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
        item = QListWidgetItem(f"{sig.name}  (start={sig.start}, len={sig.values.size})")
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
        
        save_signal_to_file(f"{res.name}.txt", res)
        QMessageBox.information(self, "Saved", f"Addition result saved to {res.name}.txt")

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
        
        save_signal_to_file(f"{res.name}.txt", res)
        QMessageBox.information(self, "Saved", f"Multiply result saved to {res.name}.txt")

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

        save_signal_to_file(f"{res.name}.txt", res)
        QMessageBox.information(self, "Saved", f"Subtraction result saved to {res.name}.txt")

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
        
        filename = f"{sig.name + "_advance_" + str(abs(k)) if k > 0 else sig.name + "_delay_" + str(abs(k))}.txt"

        res = sig.shifted(k, name=filename)
        self.add_signal_to_list(res)
        self.refresh_list_items()
        
        save_signal_to_file(filename, res)
        QMessageBox.information(self, "Saved", f"Shift result saved to {filename}")

    def fold_selected(self):
        idxs = self.get_selected_indices()
        if len(idxs) != 1:
            QMessageBox.information(self, "Select one", "Select one signal to fold.")
            return
        sig = self.signals[idxs[0]]
        res = sig.folded(name=f"{sig.name}_fold")
        self.add_signal_to_list(res)
        self.refresh_list_items()
        
        save_signal_to_file(f"{res.name}.txt", res)
        QMessageBox.information(self, "Saved", f"Folding result saved to {res.name}.txt")

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
            try:
                sig = self.signals[i]
                if delete_from_disk and os.path.exists(sig.path):
                    try:
                        os.remove(sig.path)
                    except Exception as e:
                        failed.append((sig.name, str(e)))
                del self.signals[i]
                deleted_count += 1
            except Exception as e:
                failed.append((getattr(sig, "name", f"Signal {i}"), str(e)))

        self.refresh_list_items()

        if failed:
            msg = f"{deleted_count} signal(s) removed.\n\nSome files could not be deleted:\n"
            msg += "\n".join([f"• {name}: {err}" for name, err in failed])
            QMessageBox.warning(self, "Partial Deletion", msg)
        else:
            msg_type = "and deleted from disk" if delete_from_disk else "from the list only"
            QMessageBox.information(
                self,
                "Deletion Complete",
                f"{deleted_count} signal(s) successfully removed {msg_type}."
            )

    def rename_item(self, item):
        idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        sig = self.signals[idx]
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", QLineEdit.EchoMode.Normal, sig.name)
        if ok and new_name.strip():
            sig.name = new_name.strip()
            self.refresh_list_items()


def save_signal_to_file(filename, sig: Signal):
    with open(filename, "w") as f:
     f.write(f"{len(sig.values)}\n")
     for i, v in zip(sig.indices(), sig.values):
         f.write(f"{i} {v}\n")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
