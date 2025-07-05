"""
Benchmark Window Module

This module provides the benchmark interface for the MetaBCI GUI application.
It includes components for paradigm selection, model configuration, dataset selection,
and benchmark execution control.

Features:
- Paradigm selection (SSVEP, P300, MI)
- Model selection with custom and preset options
- Dataset selection with parameter validation
- Benchmark execution with thread management
- Real-time compatibility checking
- Parameter validation and error handling

Classes:
- BenchmarkRunnerThread: Thread for running benchmark tests
- BenchmarkWidget: Main container for benchmark interface
- ParadigmSelectionCard: Paradigm selection interface
- ModelSelectionCard: Model selection and configuration
- DatasetSelectionCard: Dataset selection with parameters
- ExecutionCard: Benchmark execution control

Author: DSG
Date: 2025-07-04
"""

from PySide6.QtCore import Qt, QThread, Signal

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
)

from qfluentwidgets import (
    BodyLabel,
    PushButton,
    FluentIcon,
    ScrollArea,
    GroupHeaderCardWidget,
    ComboBox,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    CheckBox,
    IndeterminateProgressRing,
)

import resource_rc
from core.plot import MultiSelectComboBox
from benchopt import run_benchmark


class BenchmarkRunnerThread(QThread):
    """
    Benchmark test runner thread
    
    This thread handles the execution of benchmark tests in the background
    without blocking the GUI.
    """

    finished = Signal(bool, str)
    error = Signal(str)
    progress = Signal(int)  # Progress signal for updating progress bar

    def __init__(self, solver_names, dataset_names, max_runs=None):
        """
        Initialize the benchmark runner thread
        
        Args:
            solver_names (list): List of solver/algorithm names to test
            dataset_names (list): List of dataset names to use
            max_runs (int, optional): Maximum number of runs for each combination
        """
        super().__init__()
        self.solver_names = solver_names
        self.dataset_names = dataset_names
        self.max_runs = max_runs
        self.should_stop = False

    def run(self):
        """
        Execute the benchmark test
        
        This method runs in a separate thread and calls the benchopt
        library to execute the benchmark with the specified parameters.
        """
        try:
            # Use absolute path to ensure benchmark folder is found correctly
            import os
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            benchmark_path = os.path.join(current_dir, "benchmark")
            
            # Emit initial progress
            self.progress.emit(10)
            
            if not self.should_stop:
                self.progress.emit(30)
                
                # Start benchmark execution
                run_benchmark(
                    benchmark_path=benchmark_path,
                    solver_names=self.solver_names,
                    dataset_names=self.dataset_names,
                    max_runs=self.max_runs,
                )
                
                # Emit completion progress
                if not self.should_stop:
                    self.progress.emit(100)
                    self.finished.emit(True, "Benchmark test completed successfully")

        except Exception as e:
            print(f"Benchmark test exception details: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Benchmark test execution failed: {str(e)}")

    def stop(self):
        """Stop the benchmark test"""
        self.should_stop = True
        self.terminate()


class BenchmarkWidget(ScrollArea):
    """
    Main benchmark widget container
    
    This widget serves as the main container for all benchmark-related
    interface components.
    """

    def __init__(self, parent=None):
        """Initialize the benchmark widget"""
        super().__init__(parent)

        self.view = QWidget(self)
        
        self.paradigmCard = ParadigmSelectionCard(self.view)
        self.modelCard = ModelSelectionCard(self.view)
        self.datasetCard = DatasetSelectionCard(self.view)
        self.executionCard = ExecutionCard(self.view)
        
        self.paradigmCard.paradigmCombo.currentTextChanged.connect(self._onParadigmChanged)
        
        self.setupLayout()
        
        self.view.setAutoFillBackground(False)
        palette = self.view.palette()
        palette.setColor(self.view.backgroundRole(), Qt.transparent)
        self.view.setPalette(palette)

        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea{border: none; background: transparent}")
    
    def _onParadigmChanged(self, paradigm):
        """Handle paradigm change events"""
        self.modelCard.updateModelOptions(paradigm)
        self.datasetCard.updateDatasetOptions(paradigm)
        self.checkCompatibilityAndShowInfo()

    def setupLayout(self):
        """Setup the main layout"""
        Layout = QVBoxLayout(self.view)
        Layout.setContentsMargins(20, 20, 20, 20)
        Layout.setSpacing(20)
        
        Layout.addWidget(self.paradigmCard, 0, Qt.AlignTop)
        Layout.addWidget(self.modelCard, 0, Qt.AlignTop)
        Layout.addWidget(self.datasetCard, 0, Qt.AlignTop)
        Layout.addWidget(self.executionCard, 0, Qt.AlignTop)

    def checkCompatibilityAndShowInfo(self):
        """Check algorithm and dataset compatibility and show information"""
        selected_algorithms = []
        
        if self.modelCard.customModelCheck.isChecked():
            custom_text = self.modelCard.customModelEdit.text().strip()
            if custom_text:
                selected_algorithms.extend([x.strip() for x in custom_text.split(',') if x.strip()])
        
        preset_models = self.modelCard.presetModelCombo.get_selected_items()
        if preset_models:
            selected_algorithms.extend(preset_models)
        
        selected_datasets = self.datasetCard.datasetCombo.get_selected_items()
        
        if selected_datasets and selected_algorithms:
            incompatible_pairs = self.datasetCard.checkDatasetAlgorithmCompatibility(
                selected_datasets, selected_algorithms
            )
            
            if incompatible_pairs:
                warning_msg = "The following combinations are incompatible: "
                warnings = []
                for dataset, algo in incompatible_pairs:
                    warnings.append(f"{algo} algorithm does not support {dataset} dataset")
                warning_msg += "; ".join(warnings[:3])
                if len(warnings) > 3:
                    warning_msg += f" and {len(warnings)} more issues"
                
                self.showInfoBar(warning_msg, "warning")
                return False
            else:
                if hasattr(self, '_last_warning_shown') and self._last_warning_shown:
                    self.showInfoBar("Algorithm and dataset compatibility check passed", "success")
                    self._last_warning_shown = False
                return True
        
        if hasattr(self, '_last_warning_shown'):
            self._last_warning_shown = False
        return True
    
    def showInfoBar(self, message, type="info"):
        """Display information bar with message"""
        if type == "success":
            InfoBar.success(
                title="Success",
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )
        elif type == "warning":
            InfoBar.warning(
                title="Warning",
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=4000,
                parent=self,
            )
            self._last_warning_shown = True
        elif type == "error":
            InfoBar.error(
                title="Error",
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=4000,
                parent=self,
            )
        else:
            InfoBar.info(
                title="Information",
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )

    def getCurrentSettings(self):
        """
        Get all current settings
        
        This method collects all current settings from the various cards
        and returns them as a structured dictionary.
        
        Returns:
            dict: Dictionary containing all current settings
        """
        duration_values = self.datasetCard._parse_multi_dataset_array_input(self.datasetCard.durationLineEdit.text())
        subject_values = self.datasetCard._parse_multi_dataset_array_input(self.datasetCard.subjectLineEdit.text())
        
        custom_models = None
        if self.modelCard.customModelCheck.isChecked():
            custom_text = self.modelCard.customModelEdit.text().strip()
            if custom_text:
                custom_models = [x.strip() for x in custom_text.split(',') if x.strip()]
        
        settings = {
            'paradigm': self.paradigmCard.paradigmCombo.currentText(),
            'use_custom_model': self.modelCard.customModelCheck.isChecked(),
            'custom_models': custom_models,
            'preset_models': self.modelCard.presetModelCombo.get_selected_items(),
            'datasets': self.datasetCard.datasetCombo.get_selected_items(),
            'duration': duration_values,
            'subject': subject_values
        }
        return settings


class ParadigmSelectionCard(GroupHeaderCardWidget):
    """
    Paradigm selection card widget
    
    This card provides interface for selecting the experimental paradigm
    (SSVEP, P300, MI) which determines available algorithms and datasets.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the paradigm selection card
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.setTitle("Paradigm Selection")
        
        self.contentWidget = QWidget()
        self.cardLayout = QVBoxLayout(self.contentWidget)
        self.addGroup(FluentIcon.TILES, "Paradigm Selection", "Select experimental paradigm", self.contentWidget)
        
        self.paradigmCombo = ComboBox()
        self.paradigmCombo.addItems(["SSVEP", "P300", "MI"])
        self.paradigmCombo.setCurrentIndex(0)
        
        paradigmLayout = QHBoxLayout()
        paradigmLayout.addWidget(BodyLabel("Paradigm:"))
        paradigmLayout.addWidget(self.paradigmCombo)
        paradigmLayout.addStretch()
        
        self.cardLayout.addLayout(paradigmLayout)
        self.setMinimumHeight(120)


class ModelSelectionCard(GroupHeaderCardWidget):
    """
    Model selection card widget
    
    This card provides interface for selecting models/algorithms including
    both custom models and preset models. It supports dynamic model options
    based on the selected paradigm.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the model selection card
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.setTitle("Model Selection")

        self.contentWidget = QWidget()
        self.cardLayout = QVBoxLayout(self.contentWidget)
        self.addGroup(FluentIcon.APPLICATION, "Model Selection", "Select preset models or custom models", self.contentWidget)

        self.customModelCheck = CheckBox("Use custom model")
        self.cardLayout.addWidget(self.customModelCheck)

        customLayout = QHBoxLayout()
        customLayout.addWidget(BodyLabel("Algorithm names:"))
        self.customModelEdit = LineEdit()
        self.customModelEdit.setPlaceholderText("e.g., [SCCA, TRCA]")
        self.customModelEdit.setEnabled(False)
        customLayout.addWidget(self.customModelEdit)
        customLayout.addStretch()
        self.cardLayout.addLayout(customLayout)

        presetLayout = QHBoxLayout()
        presetLayout.addWidget(BodyLabel("Preset models:"))
        self.presetModelCombo = MultiSelectComboBox()
        self.presetModelCombo.setText("Please select models...")
        presetLayout.addWidget(self.presetModelCombo)
        presetLayout.addStretch()
        self.cardLayout.addLayout(presetLayout)

        self.model_data = {
            "SSVEP": [
                "SCCA", "FBSCCA", "ECCA", "FBECCA", "ItCCA", "FBItCCA",
                "TtCCA", "FBTtCCA", "MsetCCA", "FBMsetCCA", "MsetCCAR", "FBMsetCCA",
                "TRCA", "FBTRCA", "TRCAR", "FBTRCAR", "TDCA", "FBTDCA",
            ],
            "P300": ["LDA", "SKLDA", "STDA", "DCPM"],
            "MI": ["CSP", "FBCSP", "MultiCSP", "FBMultiCSP", "DSP", "FBDSP", "SSCOR", "FBSSCOR"],
        }

        self.customModelCheck.toggled.connect(self._onCustomModelToggled)
        self.customModelEdit.textChanged.connect(self._onCustomModelTextChanged)
        self.presetModelCombo.selectionChanged.connect(self._onModelSelectionChanged)
    
        self.updateModelOptions("SSVEP")
        self.setMinimumHeight(180)

    def _onCustomModelTextChanged(self):
        """Handle custom model text changes"""
        benchmark_widget = self.parent()
        while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
            benchmark_widget = benchmark_widget.parent()
        
        if benchmark_widget:
            benchmark_widget.checkCompatibilityAndShowInfo()

    def _onCustomModelToggled(self, checked):
        """Handle custom model checkbox toggle"""
        self.customModelEdit.setEnabled(checked)
        
        benchmark_widget = self.parent()
        while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
            benchmark_widget = benchmark_widget.parent()
        
        if benchmark_widget:
            benchmark_widget.checkCompatibilityAndShowInfo()

    def updateModelOptions(self, paradigm):
        """Update model options based on selected paradigm"""
        self.presetModelCombo.clear_selection()
        self.presetModelCombo.all_items.clear()
        self.presetModelCombo.checkboxes.clear()
        self.presetModelCombo.original_texts.clear()
        self.presetModelCombo.menu.clear()

        for model in self.model_data[paradigm]:
            self.presetModelCombo.addItem(model)
        self.presetModelCombo.setText("Please select models...")

    def _onModelSelectionChanged(self, selected_models):
        """Handle model selection changes"""
        benchmark_widget = self.parent()
        while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
            benchmark_widget = benchmark_widget.parent()
        
        if benchmark_widget:
            benchmark_widget.checkCompatibilityAndShowInfo()


class DatasetSelectionCard(GroupHeaderCardWidget):
    """
    Dataset selection card widget
    
    This card provides interface for selecting datasets and configuring
    their parameters (duration, subject). It includes validation for
    parameter ranges and compatibility checking with algorithms.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the dataset selection card
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.setTitle("Dataset Selection")

        self.contentWidget = QWidget()
        self.cardLayout = QVBoxLayout(self.contentWidget)
        self.addGroup(
            FluentIcon.BOOK_SHELF,
            "Dataset Selection",
            "Select datasets and configure parameters",
            self.contentWidget,
        )

        datasetLayout = QHBoxLayout()
        datasetLayout.addWidget(BodyLabel("Dataset:"))
        self.datasetCombo = MultiSelectComboBox()
        self.datasetCombo.setText("Please select datasets...")
        datasetLayout.addWidget(self.datasetCombo)
        datasetLayout.addStretch()
        self.cardLayout.addLayout(datasetLayout)

        durationLayout = QHBoxLayout()
        durationLayout.addWidget(BodyLabel("Duration:"))
        self.durationLineEdit = LineEdit()
        self.durationLineEdit.setPlaceholderText("e.g., [0.4, 0.8, 1.2]")
        durationLayout.addWidget(self.durationLineEdit)
        durationLayout.addStretch()
        self.cardLayout.addLayout(durationLayout)

        subjectLayout = QHBoxLayout()
        subjectLayout.addWidget(BodyLabel("Subject:"))
        self.subjectLineEdit = LineEdit()
        self.subjectLineEdit.setPlaceholderText("e.g., [1, 2, 3]")
        subjectLayout.addWidget(self.subjectLineEdit)
        subjectLayout.addStretch()
        self.cardLayout.addLayout(subjectLayout)

        self.dataset_data = {
            "SSVEP": ["Wang2016", "Nakanishi2015", "BETA"],
            "P300": ["Cattan_P300"],
            "MI": ["AlexMI", "munichmi", "bnci2014001", "eegbci", "schirrmeister2017", "weibo2014"],
        }

        self.dataset_configs = {
            "Wang2016": {
                "duration_range": (0.2, 5.0, 0.2),
                "subject_range": range(1, 36),
                "description": "Duration: 0.2-5.0s (step 0.2s), Subject: 1-35"
            },
            "Nakanishi2015": {
                "duration_range": (0.5, 4.0, 0.5),
                "subject_range": range(1, 11),
                "description": "Duration: 0.5-4.0s (step 0.5s), Subject: 1-10"
            },
            "BETA": {
                "duration_range": (0.2, 5.0, 0.2),
                "subject_range": range(1, 71),
                "description": "Duration: 0.2-5.0s (step 0.2s), Subject: 1-70"
            },
            "Cattan_P300": {
                "duration_range": (0.1, 1.0, 0.1),
                "subject_range": range(1, 20),
                "description": "Duration: 0.1-1.0s (step 0.1s), Subject: 1-19"
            },
            "AlexMI": {
                "duration_range": (0.5, 3.0, 0.5),
                "subject_range": range(1, 9),
                "description": "Duration: 0.5-3.0s (step 0.5s), Subject: 1-8"
            },
            "munichmi": {
                "duration_range": (0.7, 0.5, 0.1),
                "subject_range": range(1, 11),
                "description": "Duration: fixed range, Subject: 1-10"
            },
            "bnci2014001": {
                "duration_range": (2, 6, 0.2),
                "subject_range": range(1, 10),
                "description": "Duration: 2-6s (step 0.2s), Subject: 1-9",
                "incompatible_algorithms": ["CSP", "FBCSP"]
            },
            "eegbci": {
                "duration_range": (0, 3, 0.5),
                "subject_range": range(1, 110),
                "description": "Duration: 0-3s (step 0.5s), Subject: 1-109",
                "incompatible_algorithms": ["CSP", "FBCSP"]
            },
            "schirrmeister2017": {
                "duration_range": (0, 4, 0.5),
                "subject_range": range(1, 15),
                "description": "Duration: 0-4s (step 0.5s), Subject: 1-14",
                "incompatible_algorithms": ["CSP", "FBCSP"]
            },
            "weibo2014": {
                "duration_range": (3, 7, 0.5),
                "subject_range": range(1, 11),
                "description": "Duration: 3-7s (step 0.5s), Subject: 1-10",
                "incompatible_algorithms": ["CSP", "FBCSP"]
            }
        }

        self.datasetCombo.selectionChanged.connect(self._onDatasetSelectionChanged)

        self.updateDatasetOptions("SSVEP")
        self.setMinimumHeight(220)

    def updateDatasetOptions(self, paradigm):
        """Update dataset options based on selected paradigm"""
        self.datasetCombo.clear_selection()
        self.datasetCombo.all_items.clear()
        self.datasetCombo.checkboxes.clear()
        self.datasetCombo.original_texts.clear()
        self.datasetCombo.menu.clear()

        if paradigm in self.dataset_data:
            for dataset in self.dataset_data[paradigm]:
                tooltip = None
                if dataset in self.dataset_configs:
                    config = self.dataset_configs[dataset]
                    duration_range = config["duration_range"]
                    subject_range = config["subject_range"]

                    tooltip = f"Duration: {duration_range[0]}-{duration_range[1]}s (step {duration_range[2]}s)\n"
                    tooltip += f"Subject: {subject_range.start}-{subject_range.stop-1}"

                    incompatible_algos = config.get("incompatible_algorithms", [])
                    if incompatible_algos:
                        tooltip += f"\nIncompatible algorithms: {', '.join(incompatible_algos)}"

                self.datasetCombo.addItem(dataset, tooltip=tooltip)
            self.datasetCombo.setText("Please select datasets...")

    def _parse_array_input(self, text):
        """
        Parse array input, supporting [1,2,3] format or single values
        
        This method handles both array notation and single values for
        parameter input fields.
        
        Args:
            text (str): Input text to parse
            
        Returns:
            list or None: Parsed values as list, or None if invalid
        """
        if not text.strip():
            return None

        text = text.strip()

        if text.startswith('[') and text.endswith(']'):
            try:
                content = text[1:-1].strip()
                if not content:
                    return []
                values = [float(x.strip()) for x in content.split(',') if x.strip()]
                return values
            except ValueError:
                return None
        else:
            try:
                return [float(text)]
            except ValueError:
                return None

    def _parse_multi_dataset_array_input(self, text):
        """
        Parse multi-dataset array input, supporting [1,2],[2,3] or [1,2,3] format
        
        This method handles both single array format and multiple arrays for
        different datasets.
        
        Args:
            text (str): Input text to parse
            
        Returns:
            list or None: Parsed values as list of lists for multi-dataset or single list, or None if invalid
        """
        if not text.strip():
            return None

        text = text.strip()

        # Check if it's multi-dataset format like [1,2],[2,3]
        if text.count('[') > 1 and text.count(']') > 1:
            try:
                # Split multiple arrays
                arrays = []
                current_array = ""
                bracket_count = 0
                
                for char in text:
                    if char == '[':
                        bracket_count += 1
                        current_array += char
                    elif char == ']':
                        bracket_count -= 1
                        current_array += char
                        if bracket_count == 0:
                            # Completed one array
                            array_content = current_array[1:-1].strip()  # Remove brackets
                            if array_content:
                                array_values = [float(x.strip()) for x in array_content.split(',') if x.strip()]
                                arrays.append(array_values)
                            current_array = ""
                    elif bracket_count == 0 and char in ' ,':
                        # Skip spaces and commas between arrays
                        continue
                    else:
                        current_array += char
                
                return arrays if arrays else None
            except ValueError:
                return None
        else:
            # Single array format, use original method
            return self._parse_array_input(text)

    def _onDatasetSelectionChanged(self, selected_datasets):
        """Handle dataset selection changes"""
        if selected_datasets:
            if len(selected_datasets) == 1:
                # Single dataset
                dataset = selected_datasets[0]
                if dataset in self.dataset_configs:
                    config = self.dataset_configs[dataset]

                    duration_range = config["duration_range"]
                    duration_hint = f"e.g., [{duration_range[0]}, {duration_range[1]}] or {duration_range[0]} (step {duration_range[2]})"
                    self.durationLineEdit.setPlaceholderText(duration_hint)

                    subject_range = config["subject_range"]
                    subject_hint = f"e.g., [{subject_range.start}, {subject_range.stop-1}] or {subject_range.start}"
                    self.subjectLineEdit.setPlaceholderText(subject_hint)
            else:
                # Multiple datasets
                self.durationLineEdit.setPlaceholderText(f"Multi-dataset format: [1,2],[2,3] or [1,2,3] ({len(selected_datasets)} datasets)")
                self.subjectLineEdit.setPlaceholderText(f"Multi-dataset format: [1,2],[2,3] or [1,2,3] ({len(selected_datasets)} datasets)")
        else:
            self.durationLineEdit.setPlaceholderText("e.g., [0.4, 0.8, 1.2] or 1")
            self.subjectLineEdit.setPlaceholderText("e.g., [1, 2, 3] or 1")

        benchmark_widget = self.parent()
        while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
            benchmark_widget = benchmark_widget.parent()

        if benchmark_widget:
            benchmark_widget.checkCompatibilityAndShowInfo()

    def checkDatasetAlgorithmCompatibility(self, selected_datasets, selected_algorithms):
        """
        Check compatibility between datasets and algorithms
        
        This method checks if the selected algorithms are compatible
        with the selected datasets based on predefined incompatibility rules.
        
        Args:
            selected_datasets (list): List of selected dataset names
            selected_algorithms (list): List of selected algorithm names
            
        Returns:
            list: List of incompatible (dataset, algorithm) pairs
        """
        incompatible_pairs = []

        for dataset in selected_datasets:
            if dataset in self.dataset_configs:
                config = self.dataset_configs[dataset]
                incompatible_algos = config.get("incompatible_algorithms", [])

                for algo in selected_algorithms:
                    if algo in incompatible_algos:
                        incompatible_pairs.append((dataset, algo))

        return incompatible_pairs

    def validateDatasetParameters(self, selected_datasets, duration_values, subject_values):
        """
        Validate dataset parameters against valid ranges
        
        This method checks if the provided duration and subject values
        are within the valid ranges for each selected dataset.
        
        Args:
            selected_datasets (list): List of selected dataset names
            duration_values (list): List of duration values to validate (can be list of lists for multi-dataset)
            subject_values (list): List of subject values to validate (can be list of lists for multi-dataset)
            
        Returns:
            list: List of error messages for invalid parameters
        """
        errors = []

        # Handle duration parameters
        duration_per_dataset = []
        if duration_values:
            if isinstance(duration_values[0], list):
                # Multi-dataset format: [[1,2], [2,3]]
                duration_per_dataset = duration_values
            else:
                # Single array format: [1,2,3] - use same duration for all datasets
                duration_per_dataset = [duration_values] * len(selected_datasets)
        else:
            duration_per_dataset = [[]] * len(selected_datasets)

        # Handle subject parameters
        subject_per_dataset = []
        if subject_values:
            if isinstance(subject_values[0], list):
                # Multi-dataset format: [[1,2], [2,3]]
                subject_per_dataset = subject_values
            else:
                # Single array format: [1,2,3] - use same subject for all datasets
                subject_per_dataset = [subject_values] * len(selected_datasets)
        else:
            subject_per_dataset = [[]] * len(selected_datasets)

        for i, dataset in enumerate(selected_datasets):
            if dataset not in self.dataset_configs:
                continue

            config = self.dataset_configs[dataset]

            # Validate duration values
            if i < len(duration_per_dataset) and duration_per_dataset[i]:
                duration_range = config["duration_range"]
                min_dur, max_dur, step_dur = duration_range

                for dur in duration_per_dataset[i]:
                    if dur < min_dur or dur > max_dur:
                        errors.append(f"{dataset}: Duration {dur} exceeds range [{min_dur}-{max_dur}]")
                    elif step_dur > 0:
                        min_dur_int = round(min_dur * 1000)
                        dur_int = round(dur * 1000)
                        step_dur_int = round(step_dur * 1000)

                        if (dur_int - min_dur_int) % step_dur_int != 0:
                            steps = round((dur_int - min_dur_int) / step_dur_int)
                            nearest_valid = min_dur + steps * step_dur
                            errors.append(f"{dataset}: Duration {dur} doesn't match step {step_dur} (nearest valid: {nearest_valid})")

            if i < len(subject_per_dataset) and subject_per_dataset[i]:
                subject_range = config["subject_range"]
                for subj in subject_per_dataset[i]:
                    subj_int = int(subj)
                    if subj_int not in subject_range:
                        errors.append(f"{dataset}: Subject {subj_int} exceeds range [{subject_range.start}-{subject_range.stop-1}]")

        return errors

    def getValidDurationValues(self, dataset):
        """
        Get valid duration values for a dataset
        
        This method generates a list of all valid duration values
        for the specified dataset based on its configuration.
        
        Args:
            dataset (str): Dataset name
            
        Returns:
            list: List of valid duration values
        """
        if dataset not in self.dataset_configs:
            return []

        config = self.dataset_configs[dataset]
        min_dur, max_dur, step_dur = config["duration_range"]

        if step_dur <= 0:
            return []

        valid_values = []
        current = min_dur
        while current <= max_dur + 1e-6:
            valid_values.append(round(current, 1))
            current += step_dur

        return valid_values

    def getValidSubjectValues(self, dataset):
        """
        Get valid subject values for a dataset
        
        This method returns a list of all valid subject values
        for the specified dataset.
        
        Args:
            dataset (str): Dataset name
            
        Returns:
            list: List of valid subject values
        """
        if dataset not in self.dataset_configs:
            return []

        config = self.dataset_configs[dataset]
        subject_range = config["subject_range"]
        return list(subject_range)


class ExecutionCard(GroupHeaderCardWidget):
    """
    Execution control card widget
    
    This card provides interface for controlling benchmark execution
    including start/stop buttons and parameter validation.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the execution card
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.setTitle("Execution Control")

        self.contentWidget = QWidget()
        self.cardLayout = QVBoxLayout(self.contentWidget)
        self.addGroup(FluentIcon.SPEED_HIGH, "Start Benchmark Test", "", self.contentWidget)

        # Create container for buttons and progress bar
        self.controlContainer = QWidget()
        self.controlLayout = QHBoxLayout(self.controlContainer)
        self.controlLayout.setContentsMargins(0, 0, 0, 0)
        self.controlLayout.setSpacing(8)

        # Buttons
        self.startButton = PushButton("Start Benchmark", icon=FluentIcon.PLAY)
        self.stopButton = PushButton("Stop Benchmark", icon=FluentIcon.PAUSE)
        self.stopButton.setEnabled(False)

        # Progress bar
        self.progressBar = IndeterminateProgressRing()
        self.progressBar.setFixedSize(20, 20)
        self.progressBar.setVisible(False)
        self.progressBar.setValue(0)

        # Add to control layout
        self.controlLayout.addWidget(self.startButton)
        self.controlLayout.addWidget(self.stopButton)
        self.controlLayout.addWidget(self.progressBar)
        self.controlLayout.addStretch()

        # Add control container to card layout
        self.cardLayout.addWidget(self.controlContainer)

        self.startButton.clicked.connect(self._onStartClicked)
        self.stopButton.clicked.connect(self._onStopClicked)

        self.benchmark_thread = None

        self.setMinimumHeight(120)

    def _onStartClicked(self):
        """Handle start button click"""
        validation_result = self._validateInputs()
        if not validation_result['valid']:
            benchmark_widget = self.parent()
            while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
                benchmark_widget = benchmark_widget.parent()

            if benchmark_widget:
                benchmark_widget.showInfoBar(validation_result['message'], "error")
            return

        benchmark_params = self._generateBenchmarkParams()
        if not benchmark_params:
            benchmark_widget = self.parent()
            while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
                benchmark_widget = benchmark_widget.parent()

            if benchmark_widget:
                benchmark_widget.showInfoBar("Failed to generate benchmark parameters", "error")
            return

        # Update UI state - show progress bar and disable start button
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        
        benchmark_widget = self.parent()
        while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
            benchmark_widget = benchmark_widget.parent()

        if benchmark_widget:
            benchmark_widget.showInfoBar("Benchmark test started running...", "info")

        # Create and configure benchmark thread
        self.benchmark_thread = BenchmarkRunnerThread(
            solver_names=benchmark_params['solver_names'],
            dataset_names=benchmark_params['dataset_names'],
            max_runs=benchmark_params['max_runs']
        )
        
        # Connect signals
        self.benchmark_thread.finished.connect(self._onBenchmarkFinished)
        self.benchmark_thread.error.connect(self._onBenchmarkError)
        self.benchmark_thread.progress.connect(self._onProgressUpdate)
        
        # Start the thread
        self.benchmark_thread.start()

    def _onStopClicked(self):
        """Handle stop button click"""
        if self.benchmark_thread and self.benchmark_thread.isRunning():
            self.benchmark_thread.stop()
            self.benchmark_thread.wait()
            
            benchmark_widget = self.parent()
            while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
                benchmark_widget = benchmark_widget.parent()

            if benchmark_widget:
                benchmark_widget.showInfoBar("Benchmark test stopped", "info")
        
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        # Hide progress bar when benchmark is stopped
        self.progressBar.setVisible(False)

    def _validateInputs(self):
        """
        Validate all input parameters
        
        This method performs comprehensive validation of all input parameters
        before starting the benchmark execution.
        
        Returns:
            dict: Dictionary with 'valid' boolean and 'message' string
        """
        benchmark_widget = self.parent()
        while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
            benchmark_widget = benchmark_widget.parent()

        if not benchmark_widget:
            return {'valid': False, 'message': 'Unable to find benchmark component'}

        settings = benchmark_widget.getCurrentSettings()

        if not settings['paradigm']:
            return {'valid': False, 'message': 'Please select a paradigm'}

        selected_algorithms = []
        if settings['use_custom_model'] and settings['custom_models']:
            selected_algorithms.extend(settings['custom_models'])
        if settings['preset_models']:
            selected_algorithms.extend(settings['preset_models'])
        
        if not selected_algorithms:
            return {'valid': False, 'message': 'Please select preset models or enter custom model names'}

        if not settings['datasets']:
            return {'valid': False, 'message': 'Please select datasets'}

        if settings['duration'] is None:
            return {'valid': False, 'message': 'Duration format error, please use list format like [0.4, 0.8] or multi-dataset format [1,2],[2,3]'}

        if settings['subject'] is None:
            return {
                "valid": False,
                "message": "Subject format error, please use list format like [1, 2, 3] or multi-dataset format [1,2],[2,3]",
            }

        incompatible_pairs = benchmark_widget.datasetCard.checkDatasetAlgorithmCompatibility(
            settings['datasets'], selected_algorithms
        )

        if incompatible_pairs:
            incompatible_info = []
            for dataset, algo in incompatible_pairs:
                incompatible_info.append(f"{algo} algorithm does not support {dataset} dataset")
            return {'valid': False, 'message': '; '.join(incompatible_info)}

        parameter_errors = benchmark_widget.datasetCard.validateDatasetParameters(
            settings['datasets'], settings['duration'], settings['subject']
        )

        if parameter_errors:
            return {'valid': False, 'message': '; '.join(parameter_errors[:3])}

        return {'valid': True, 'message': 'Validation passed'}

    def _onBenchmarkFinished(self, success, message):
        """Handle benchmark completion"""
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        # Hide progress bar when benchmark completes
        self.progressBar.setVisible(False)
        
        benchmark_widget = self.parent()
        while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
            benchmark_widget = benchmark_widget.parent()

        if benchmark_widget:
            if success:
                benchmark_widget.showInfoBar(message, "success")
            else:
                benchmark_widget.showInfoBar(f"Benchmark test failed: {message}", "error")
    
    def _onBenchmarkError(self, error_message):
        """Handle benchmark error"""
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        # Hide progress bar when benchmark errors
        self.progressBar.setVisible(False)
        
        benchmark_widget = self.parent()
        while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
            benchmark_widget = benchmark_widget.parent()

        if benchmark_widget:
            benchmark_widget.showInfoBar(error_message, "error")
    
    def _onProgressUpdate(self, value):
        """Handle progress updates from benchmark thread"""
        self.progressBar.setValue(value)
    
    def _generateBenchmarkParams(self):
        """
        Generate benchmark parameters
        
        This method generates the parameters needed for benchmark execution
        based on current settings and paradigm-specific requirements.
        
        Returns:
            dict or None: Dictionary with benchmark parameters, or None if failed
        """
        benchmark_widget = self.parent()
        while benchmark_widget and not isinstance(benchmark_widget, BenchmarkWidget):
            benchmark_widget = benchmark_widget.parent()

        if not benchmark_widget:
            return None

        settings = benchmark_widget.getCurrentSettings()
        
        # Determine max_runs based on paradigm
        max_runs_map = {
            'P300': 11,
            'MI': 4,
            'SSVEP': None
        }
        max_runs = max_runs_map.get(settings['paradigm'], None)
        
        solver_names = []
        paradigm = settings['paradigm']
        
        if settings['use_custom_model'] and settings['custom_models']:
            for algo in settings['custom_models']:
                if paradigm == 'SSVEP':
                    solver_name = f"{paradigm}-docomposition-algo[custom_model=[{algo}], model=None, module_name=algorithm, padding_len=None]"
                else:
                    solver_name = f"{paradigm}-docomposition-algo[custom_model=[{algo}], model=None, module_name=algorithm]"
                solver_names.append(solver_name)
        
        if settings['preset_models']:
            for algo in settings['preset_models']:
                if paradigm == 'SSVEP':
                    solver_name = f"{paradigm}-docomposition-algo[model=[{algo}], module_name=decomposition, padding_len=0]"
                else:
                    solver_name = f"{paradigm}-docomposition-algo[model=[{algo}], module_name=decomposition]"
                solver_names.append(solver_name)
        
        dataset_names = []
        selected_datasets = settings['datasets']
        
        # Handle duration - support multi-dataset format
        duration_per_dataset = []
        if settings['duration']:
            if isinstance(settings['duration'][0], list):
                # Multi-dataset format: [[1,2], [2,3]]
                duration_per_dataset = settings['duration']
                # Ensure duration arrays match dataset count
                if len(duration_per_dataset) < len(selected_datasets):
                    # If duration arrays are fewer than datasets, use the last one to fill
                    last_duration = duration_per_dataset[-1] if duration_per_dataset else [1.0]
                    while len(duration_per_dataset) < len(selected_datasets):
                        duration_per_dataset.append(last_duration)
            else:
                # Single array format: [1,2,3] - use same duration for all datasets
                single_duration = settings['duration']
                duration_per_dataset = [single_duration] * len(selected_datasets)
        else:
            # No duration, use default values
            duration_per_dataset = [[1.0]] * len(selected_datasets)
        
        # Handle subject - support multi-dataset format
        subject_per_dataset = []
        if settings['subject']:
            if isinstance(settings['subject'][0], list):
                # Multi-dataset format: [[1,2], [2,3]]
                subject_per_dataset = settings['subject']
                # Ensure subject arrays match dataset count
                if len(subject_per_dataset) < len(selected_datasets):
                    # If subject arrays are fewer than datasets, use the last one to fill
                    last_subject = subject_per_dataset[-1] if subject_per_dataset else [1]
                    while len(subject_per_dataset) < len(selected_datasets):
                        subject_per_dataset.append(last_subject)
            else:
                # Single array format: [1,2,3] - use same subject for all datasets
                single_subject = settings['subject']
                subject_per_dataset = [single_subject] * len(selected_datasets)
        else:
            # No subject, use default values
            subject_per_dataset = [[1]] * len(selected_datasets)
        
        for i, dataset in enumerate(selected_datasets):
            params = []
            
            if dataset in ['Wang2016', 'BETA']:
                params.append("channel=occipital_9")
            elif dataset in ['Nakanishi2015']:
                params.append("channel=occipital_8")
            
            if i < len(duration_per_dataset):
                duration_values = duration_per_dataset[i]
                formatted_duration = []
                for val in duration_values:
                    if val == int(val):
                        formatted_duration.append(str(int(val)))
                    else:
                        formatted_duration.append(str(val))
                duration_str = '[' + ','.join(formatted_duration) + ']'
                params.append(f"duration={duration_str}")
            
            if i < len(subject_per_dataset):
                subject_values = subject_per_dataset[i]
                formatted_subject = []
                for val in subject_values:
                    formatted_subject.append(str(int(val)))
                subject_str = '[' + ','.join(formatted_subject) + ']'
                params.append(f"subject={subject_str}")
            
            if params:
                dataset_name = f"{dataset}[{','.join(params)}]"
            else:
                dataset_name = dataset
            
            dataset_names.append(dataset_name)
        
        return {
            'solver_names': solver_names,
            'dataset_names': dataset_names,
            'max_runs': max_runs
        }
