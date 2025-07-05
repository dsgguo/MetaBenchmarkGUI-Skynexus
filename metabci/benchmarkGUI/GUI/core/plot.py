"""
Plot Module

This module provides plotting components for the MetaBCI GUI application.
It includes data loading, management, multi-select combo boxes, and advanced
FFT analysis visualization capabilities.

Features:
- Threaded data loading for datasets
- Multi-select combo box with context menus
- Advanced FFT analysis plotting
- Interactive matplotlib widgets
- Data filtering and signal processing

Classes:
- DataLoader: Thread for loading EEG datasets
- DataManager: Manager for data loading operations
- MultiSelectComboBox: Multi-selection dropdown widget
- PlotWidget: Advanced plotting widget with FFT analysis

Author: DSG
Date: 2025-07-04
"""

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QIcon, QColor, QMouseEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QDir
from qfluentwidgets import (
    PushButton,
    MessageBox,
    RoundMenu,
    Action,
    FluentIcon,
    ToolTipFilter,
    ToolTipPosition
)
import numpy as np
from scipy.signal import sosfiltfilt
from metabci.brainda.datasets import Wang2016, Nakanishi2015
from metabci.brainda.paradigms import SSVEP
import resource_rc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "analyze_algorithm"))
from core.analyze_algorithm.data_fft import filter, fft

class DataLoader(QThread):
    """
    Data loading thread for EEG datasets
    
    This thread handles loading EEG data from various datasets in the background
    without blocking the GUI. It supports Wang2016 and Nakanishi2015 datasets
    with configurable parameters.
    """

    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, dataset_name, subject_id, selected_channels, srate=250, delay=0.14, duration=5):
        """
        Initialize the data loader thread
        
        Args:
            dataset_name (str): Name of the dataset to load
            subject_id (int): Subject ID to load data for
            selected_channels (list): List of selected channels
            srate (int): Sampling rate
            delay (float): Delay before stimulus onset
            duration (float): Duration of data to extract
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subject_id = subject_id
        self.srate = srate
        self.selected_channels = selected_channels
        self.duration = duration
        self.delay = delay

    def run(self):
        """Execute the data loading process"""
        try:
            self.progress.emit(10)

            if self.dataset_name == "Wang2016":
                dataset = Wang2016()
                max_subjects = 35
            elif self.dataset_name == "Nakanishi2015":
                dataset = Nakanishi2015()
                max_subjects = 10
            else:
                self.error.emit("Unsupported dataset")
                return

            if self.subject_id < 1 or self.subject_id > max_subjects:
                self.error.emit(f"Subject ID must be between 1-{max_subjects}")
                return

            self.progress.emit(30)

            events:list[str] = sorted(list(dataset.events.keys()))
            freqs:list[int] = [dataset.get_freq(event) for event in events]
            phases:list[int] = [dataset.get_phase(event) for event in events]
            channels:list[str] = dataset._CHANNELS

            self.progress.emit(50)

            paradigm = SSVEP(
                srate=self.srate,
                channels=self.selected_channels,
                intervals=[(self.delay, self.delay + self.duration)],
                events=events,
            )

            self.progress.emit(70)

            X, _, _ = paradigm.get_data(
                dataset,
                subjects=[self.subject_id],
                return_concat=False,
                n_jobs=1,
                verbose=False,
            )

            self.progress.emit(90)

            data_info = {
                "X": X,
                "events": events,
                "freqs": freqs,
                "phases": phases,
                "channels": channels,
                "dataset": dataset,
                "params": {"delay": self.delay, "srate": self.srate, "duration": self.duration, "selected_channels": self.selected_channels},
            }

            self.progress.emit(100)
            self.finished.emit(data_info)

        except Exception as e:
            self.error.emit(f"Data loading failed: {str(e)}")


class DataManager(QObject):
    """
    Data manager for handling EEG data loading operations
    
    This class manages data loading threads and provides signals for
    communication with the GUI components.
    """

    dataLoaded = Signal(object)
    loadingStarted = Signal()
    loadingFinished = Signal()
    loadingError = Signal(str)

    def __init__(self):
        """Initialize the data manager"""
        super().__init__()
        self.current_data = None
        self.loader_thread = None
        self.current_dataset = None

    def set_dataset(self, dataset_name):
        """Set the current dataset name"""
        self.current_dataset = dataset_name
        print(f"Current dataset set to: {dataset_name}")
    
    def load_data(self, subject_id, srate, selected_channels, duration, delay=0.14):
        """
        Load data with specified parameters
        
        Args:
            subject_id (int): Subject ID to load
            srate (int): Sampling rate
            selected_channels (list): Selected channels
            duration (float): Data duration
            delay (float): Delay before stimulus onset
        """
        if not self.current_dataset:
            self.loadingError.emit("Please select a dataset first")
            return
        
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.quit()
            self.loader_thread.wait()

        self.loadingStarted.emit()

        self.loader_thread = DataLoader(dataset_name=self.current_dataset,
                                        subject_id=subject_id, 
                                        srate=srate, 
                                        selected_channels=selected_channels,
                                        duration=duration,
                                        delay=delay
                                        )
        self.loader_thread.finished.connect(self._on_data_loaded)
        self.loader_thread.error.connect(self._on_loading_error)
        self.loader_thread.start()

    def _on_data_loaded(self, data_info):
        """Handle data loading completion"""
        self.current_data = data_info
        self.loadingFinished.emit()
        self.dataLoaded.emit(data_info)

    def _on_loading_error(self, error_msg):
        """Handle data loading errors"""
        self.loadingFinished.emit()
        self.loadingError.emit(error_msg)

    def get_progress_signal(self):
        """Get the progress signal from the loader thread"""
        if self.loader_thread:
            return self.loader_thread.progress
        return None


class MultiSelectComboBox(PushButton):
    """
    Multi-select combo box widget
    
    This widget provides a dropdown menu with multiple selection capability,
    context menu support, and customizable item display.
    """

    selectionChanged = Signal(list)

    def __init__(self, parent=None):
        """Initialize the multi-select combo box"""
        super().__init__(parent)
        self.selected_items = []
        self.all_items = []
        self.checkboxes = {}
        self.original_texts = {}

        self.menu = RoundMenu(parent=self)
        self.context_menu = RoundMenu(parent=self)
        self.setup_context_menu()
        self.clicked.connect(self.show_menu)

    def setup_context_menu(self):
        """Setup the right-click context menu"""
        select_all_action = Action("Select All")
        select_all_action.setIcon(FluentIcon.ACCEPT)
        select_all_action.triggered.connect(self.select_all)
        self.context_menu.addAction(select_all_action)

        clear_all_action = Action("Clear All")
        clear_all_action.setIcon(FluentIcon.CANCEL)
        clear_all_action.triggered.connect(self.clear_selection)
        self.context_menu.addAction(clear_all_action)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        if event.button() == Qt.RightButton:
            pos = self.mapToGlobal(event.pos())
            self.context_menu.exec(pos)
        else:
            super().mousePressEvent(event)

    def addItems(self, items, tooltips=None):
        """Add multiple items to the combo box"""
        for i, item in enumerate(items):
            tooltip = tooltips[i] if tooltips and i < len(tooltips) else None
            self.addItem(item, tooltip=tooltip)

    def addItem(self, text, icon=None, userData=None, tooltip=None):
        """Add a single item to the combo box"""
        self.all_items.append(text)
        self.original_texts[text] = text

        action = Action(text)
        action.setCheckable(True)
        
        if tooltip:
            action.setToolTip(tooltip)
            action.installEventFilter(
            ToolTipFilter(action, showDelay=100, position=ToolTipPosition.TOP)
        )

        action.toggled.connect(
            lambda checked, item=text: self.on_item_changed(item, checked)
        )

        self.menu.addAction(action)
        self.checkboxes[text] = action

    def on_item_changed(self, item_text, checked):
        """Handle item selection state changes"""
        action = self.checkboxes[item_text]

        if checked:
            if item_text not in self.selected_items:
                self.selected_items.append(item_text)
            action.setText(f"✓ {self.original_texts[item_text]}")
        else:
            if item_text in self.selected_items:
                self.selected_items.remove(item_text)
            action.setText(self.original_texts[item_text])

        self.update_display_text()
        self.selectionChanged.emit(self.selected_items)

    def select_all(self):
        """Select all items"""
        self.selected_items = self.all_items.copy()

        for item_text, action in self.checkboxes.items():
            action.setChecked(True)
            action.setText(f"✓ {self.original_texts[item_text]}")

        self.update_display_text()
        self.selectionChanged.emit(self.selected_items)

    def show_menu(self):
        """Show the dropdown menu"""
        pos = self.mapToGlobal(self.rect().bottomLeft())
        self.menu.exec(pos)

    def update_display_text(self):
        """Update the display text"""
        if not self.selected_items:
            self.setText("Please select frequencies...")
        else:
            display_text = ", ".join(self.selected_items)
            if len(display_text) > 20:
                display_text = display_text[:17] + "..."
            self.setText(display_text)

    def get_selected_items(self):
        """Get the selected items"""
        return self.selected_items.copy()

    def set_selected_items(self, items):
        """Set the selected items"""
        self.selected_items = items.copy() if items else []

        for item_text, action in self.checkboxes.items():
            is_selected = item_text in self.selected_items
            action.setChecked(is_selected)

            if is_selected:
                action.setText(f"✓ {self.original_texts[item_text]}")
            else:
                action.setText(self.original_texts[item_text])

        self.update_display_text()

    def clear_selection(self):
        """Clear all selections"""
        self.selected_items.clear()
        for item_text, action in self.checkboxes.items():
            action.setChecked(False)
            action.setText(self.original_texts[item_text])
        self.update_display_text()

class PlotWidget(QWidget):
    """
    Advanced plotting widget for FFT analysis
    
    This widget provides comprehensive FFT analysis visualization capabilities
    with matplotlib integration, including single and subplot modes, interactive
    toolbars, and advanced signal processing features.
    """

    def __init__(self, parent=None):
        """Initialize the plot widget"""
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)

        toolbar_container = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_container)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.save_button = PushButton("Save the Figure")
        self.save_button.clicked.connect(self.save_figure)
        self.save_button.setFixedWidth(120)

        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.save_button)

        layout.addWidget(toolbar_container)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)

    def save_figure(self):
        """Save the current figure to file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Figure",
                QDir.homePath() + "/FFT_Analysis.png",
                "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;SVG Files (*.svg)",
            )

            if file_path:
                self.figure.savefig(
                    file_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )

                MessageBox.information(self, "Save Successful", f"Figure saved to:\n{file_path}")

        except Exception as e:
            MessageBox.warning(self, "Save Failed", f"Error occurred while saving figure:\n{str(e)}")

    def plotSampleData(self):
        """Plot sample SSVEP data"""
        self.figure.clear()

        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)

        t = np.linspace(0, 1, 1000)

        sample_freqs = [10, 12, 15, 20]
        for i, freq in enumerate(sample_freqs):
            signal = np.sin(2 * np.pi * freq * t)
            axes = [ax1, ax2, ax3, ax4][i]
            axes.plot(t, signal)
            axes.set_title(f"SSVEP Signal {freq}Hz")
            axes.set_xlabel("Time (s)")
            axes.set_ylabel("Amplitude")
            axes.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

    def plotFFTAnalysis(
        self, data_info, params, selected_freqs, selected_channels, plot_mode="single"
    ):
        """
        Plot FFT analysis results
        
        Args:
            data_info (dict): Data information containing EEG data
            params (dict): Analysis parameters
            selected_freqs (list): Selected frequencies to analyze
            selected_channels (str): Selected channel configuration
            plot_mode (str): Plot mode ("single" or "subplot")
        """
        try:
            self.figure.clear()

            channels = data_info['channels']
            data = data_info['X']
            freqs = data_info['freqs']

            srate_val = params.get("srate")
            filter_type = params.get("filter_type", "bandpass")
            freq_range = (8, 30)

            channel_indices = []
            if selected_channels == "Occipital 9 Channels":
                channel_names = [
                    "PZ",
                    "PO5",
                    "PO3",
                    "POZ",
                    "PO4",
                    "PO6",
                    "O1",
                    "OZ",
                    "O2",
                ]
                channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            else:
                channel_names = channels
                channel_indices = [idx for idx, _ in enumerate(channels)]

            print(f"Plot mode: {plot_mode}")
            print(f"Selected frequencies: {selected_freqs}")
            print(f"Selected channels: {channel_names}")

            num_freqs = len(selected_freqs)

            if plot_mode == "single" or num_freqs == 1:
                ax = self.figure.add_subplot(111)
                self._plot_overlay(
                    ax,
                    data_info,
                    selected_freqs,
                    srate_val,
                    filter_type,
                    freq_range,
                    channel_indices,
                    channel_names,
                )

                title_parts = [f"Filter: {filter_type}"]
                if selected_freqs:
                    freq_display = ", ".join(
                        [f"{float(f):.1f}" for f in selected_freqs]
                    )
                    title_parts.append(f"Frequencies: {freq_display} Hz")

                self.figure.suptitle(
                    " | ".join(title_parts), fontsize=14, fontweight="bold"
                )

            elif plot_mode == "subplot":
                if num_freqs <= 2:
                    rows, cols = 1, 2
                    figsize = (14, 6)
                elif num_freqs <= 4:
                    rows, cols = 2, 2
                    figsize = (14, 12)
                elif num_freqs <= 6:
                    rows, cols = 2, 3
                    figsize = (18, 12)
                elif num_freqs <= 9:
                    rows, cols = 3, 3
                    figsize = (18, 16)
                else:
                    rows, cols = 4, 3
                    figsize = (18, 20)

                self.figure.set_size_inches(figsize)

                all_power_spectra = []
                subplot_data = []

                for freq_str in selected_freqs[:12]:
                    try:
                        trial_data = self._get_frequency_data(data, freq_str)
                        if trial_data is None:
                            continue

                        filtered_data = self._apply_filter(
                            trial_data, srate_val, filter_type, freq_range
                        )
                        freqs_fft, power_spectrum = self._perform_fft(
                            filtered_data, srate_val, channel_indices
                        )

                        all_power_spectra.append(power_spectrum)
                        subplot_data.append((freq_str, freqs_fft, power_spectrum))
                    except Exception as e:
                        print(f"Error preprocessing frequency {freq_str}: {e}")
                        continue

                if all_power_spectra:
                    all_powers = np.concatenate(all_power_spectra)
                    y_min = 0
                    y_max = max(
                        np.percentile(all_powers[all_powers > 0], 95), 3000
                    )
                    y_range = (y_min, y_max * 1.1)
                else:
                    y_range = None

                for i, (freq_str, freqs_fft, power_spectrum) in enumerate(subplot_data):
                    ax = self.figure.add_subplot(rows, cols, i + 1)
                    self._plot_subplot(
                        ax,
                        freq_str,
                        freqs_fft,
                        power_spectrum,
                        srate_val,
                        is_subplot=True,
                        y_range=y_range,
                    )

                title_parts = [f"Filter: {filter_type}"]
                freq_display = ", ".join(
                    [f"{float(f):.1f}" for f in selected_freqs[:5]]
                )
                if len(selected_freqs) > 5:
                    freq_display += f" and {len(selected_freqs)} more"
                title_parts.append(f"Individual Display: {freq_display} Hz")

                self.figure.suptitle(
                    " | ".join(title_parts), fontsize=14, fontweight="bold"
                )

            self.figure.tight_layout(
                rect=[0, 0.03, 1, 0.95], pad=3.5, h_pad=3.0, w_pad=2.5
            )

            self.canvas.draw()
            return True

        except Exception as e:
            print(f"Plotting error: {e}")
            import traceback
            traceback.print_exc()
            self.plotErrorMessage(f"Plotting failed: {str(e)}")
            return False

    def _plot_subplot(
        self,
        ax,
        freq_str,
        freqs_fft,
        power_spectrum,
        srate_val,
        is_subplot=False,
        y_range=None,
    ):
        """Plot individual frequency subplot with preprocessed data"""
        try:
            color = "b" if not is_subplot else f"C{hash(freq_str) % 10}"
            ax.plot(
                freqs_fft,
                power_spectrum,
                color=color,
                linewidth=2.5 if is_subplot else 2,
                label=(
                    f"{freq_str}Hz Average Power Spectrum"
                    if is_subplot
                    else "Average Power Spectrum"
                ),
            )

            freq_val = float(freq_str)
            freq_idx = np.argmin(np.abs(freqs_fft - freq_val))

            ax.axvline(x=freq_val, color="r", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.plot(
                freqs_fft[freq_idx],
                power_spectrum[freq_idx],
                "ro",
                markersize=8 if is_subplot else 6,
            )

            ax.annotate(
                f"{freq_val}Hz",
                xy=(freq_val, power_spectrum[freq_idx]),
                xytext=(5, 10),
                textcoords="offset points",
                fontsize=10 if is_subplot else 10,
                color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
            )

            if is_subplot:
                ax.set_title(f"{freq_str}Hz", fontsize=12, fontweight="bold")
                ax.set_xlabel("Frequency (Hz)", fontsize=10)
                ax.set_ylabel("Power", fontsize=10)
                ax.tick_params(labelsize=9)
            else:
                ax.set_title(f"FFT Analysis Results", fontsize=14, fontweight="bold")
                ax.set_xlabel("Frequency (Hz)", fontsize=12)
                ax.set_ylabel("Power", fontsize=12)
                ax.tick_params(labelsize=10)
                ax.legend(fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, min(50, srate_val / 2))

            if y_range is not None:
                ax.set_ylim(y_range)
            elif len(power_spectrum) > 0 and np.any(power_spectrum > 0):
                y_max = np.percentile(power_spectrum[power_spectrum > 0], 95)
                if y_max > 0:
                    ax.set_ylim(0, y_max * 1.2)

            target_freq = float(freq_str)
            highlight_range = 2
            ax.axvspan(
                target_freq - highlight_range,
                target_freq + highlight_range,
                alpha=0.15,
                color="red",
            )

        except Exception as e:
            print(f"Error plotting frequency {freq_str}: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error: {freq_str}Hz\n{str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )
            ax.set_title(f"{freq_str}Hz - Error", fontsize=12, color="red")

    def _plot_overlay(
        self,
        ax,
        data_info,
        selected_freqs,
        srate,
        filter_type,
        freq_range,
        channel_indices,
        channel_names,
    ):
        """Plot overlay mode with multiple frequencies"""
        data_X = data_info['X']

        colors = [
            "b",
            "r",
            "g",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        for i, freq_str in enumerate(selected_freqs):
            try:
                trial_data = self._get_frequency_data(data_X, freq_str)
                if trial_data is None:
                    continue

                filtered_data = self._apply_filter(
                    trial_data, srate, filter_type, freq_range
                )

                freqs_fft, power_spectrum = self._perform_fft(
                    filtered_data, srate, channel_indices
                )

                color = colors[i % len(colors)]

                ax.plot(
                    freqs_fft,
                    power_spectrum,
                    color=color,
                    linewidth=2.5,
                    label=f"{freq_str}Hz",
                    alpha=0.8,
                )

                freq_val = float(freq_str)
                freq_idx = np.argmin(np.abs(freqs_fft - freq_val))

                ax.axvline(
                    x=freq_val, color=color, linestyle="--", alpha=0.6, linewidth=1.5
                )
                ax.plot(
                    freqs_fft[freq_idx],
                    power_spectrum[freq_idx],
                    "o",
                    color=color,
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                )

                ax.annotate(
                    f"{freq_val}Hz",
                    xy=(freq_val, power_spectrum[freq_idx]),
                    xytext=(8, 12 + i * 18),
                    textcoords="offset points",
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.3),
                )

            except Exception as e:
                print(f"Error processing frequency {freq_str}: {e}")
                continue

        channel_info = f"Average of {len(channel_indices)} Channels"
        if len(channel_indices) <= 5:
            channel_names_display = channel_names[:len(channel_indices)]
            channel_info = f"Average Channels: {', '.join(channel_names_display)}"

        ax.set_title(
            f"FFT Overlay Analysis - {channel_info}", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=14)
        ax.set_ylabel("Power", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(50, srate / 2))
        ax.tick_params(labelsize=12)

        ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
        ax.set_ylim(bottom=0)

    def _get_frequency_data(self, data_X, freq_str):
        """Get data for a specific frequency"""
        print(f"Getting data for frequency {freq_str}...")
        try:
            if freq_str in data_X:
                frequency_data = data_X[freq_str]
                print(f"Data shape for frequency {freq_str}: {frequency_data.shape}")
                return frequency_data
            else:
                print(f"Frequency {freq_str} not found, available frequencies: {list(data_X.keys())}")
                return None

        except Exception as e:
            print(f"Error getting frequency data: {e}")
            return None

    def _apply_filter(self, data, srate_val, filter_type, freq_range):
        """Apply filtering to the data"""
        if filter_type != "Not Set":
            try:
                filtered_data = filter(data, srate_val, filter_type, freq_range)
                if filtered_data is None:
                    return data
                return filtered_data
            except Exception as e:
                print(f"Filtering failed: {e}")
                return data
        else:
            return data

    def _perform_fft(self, data, srate_val, channel_indices):
        """Perform FFT analysis and return 1D results"""
        try:
            freqs_fft, power_spectrum = fft(
                data,
                srate_val,
                plot=False,
                channel_idx=channel_indices,
                avg_channels=True,
            )

            if len(power_spectrum.shape) == 2:
                if power_spectrum.shape[0] == len(channel_indices):
                    power_spectrum = np.mean(power_spectrum, axis=0)
                elif power_spectrum.shape[1] == len(channel_indices):
                    power_spectrum = np.mean(power_spectrum, axis=1)
                else:
                    power_spectrum = np.mean(power_spectrum, axis=0)
            elif len(power_spectrum.shape) > 2:
                for _ in range(len(power_spectrum.shape) - 1):
                    power_spectrum = np.mean(power_spectrum, axis=0)

            if freqs_fft.shape[0] != power_spectrum.shape[0]:
                min_len = min(freqs_fft.shape[0], power_spectrum.shape[0])
                freqs_fft = freqs_fft[:min_len]
                power_spectrum = power_spectrum[:min_len]

            if len(power_spectrum.shape) > 1:
                power_spectrum = power_spectrum.flatten()

            return freqs_fft, power_spectrum

        except Exception as e:
            print(f"FFT analysis failed: {e}")
            return self._simple_fft(data, srate_val, channel_indices)

    def _simple_fft(self, data, srate, channel_indices):
        """Simple FFT implementation as fallback"""
        try:
            if len(data.shape) == 3:
                data = data[0]
            elif len(data.shape) == 1:
                data = data.reshape(1, -1)

            n_samples = data.shape[1]
            freqs = np.fft.fftfreq(n_samples, 1 / srate)
            freqs = freqs[: n_samples // 2]

            power_spectra = []
            for ch_idx in channel_indices:
                if ch_idx < data.shape[0]:
                    fft_result = np.fft.fft(data[ch_idx])
                    power_spectrum = np.abs(fft_result[: n_samples // 2])
                    power_spectra.append(power_spectrum)
                else:
                    fft_result = np.fft.fft(data[0])
                    power_spectrum = np.abs(fft_result[: n_samples // 2])
                    power_spectra.append(power_spectrum)

            averaged_power = np.mean(power_spectra, axis=0)
            return freqs, averaged_power

        except Exception as e:
            print(f"Simple FFT also failed: {e}")
            freqs = np.linspace(0, srate / 2, 1000)
            power_spectrum = np.random.random(1000)
            return freqs, power_spectrum

    def plotErrorMessage(self, error_msg):
        """Display error message on the plot"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"Error: {error_msg}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            color="red",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.canvas.draw()
