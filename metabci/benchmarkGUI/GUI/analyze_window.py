"""
Analysis Window Module

This module provides the main analysis interface for EEG data processing and visualization.
It includes components for parameter settings, data loading, and frequency domain analysis.

Author: DSG
Date: 2025-07-03
"""

from PySide6.QtWidgets import (
    QWidget, 
    QHBoxLayout,
    QVBoxLayout,
    QTableWidgetItem,
    QHeaderView,
    )
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from qfluentwidgets import (
    PushButton,
    ComboBox,
    TableWidget,
    GroupHeaderCardWidget,
    ScrollArea,
    MessageBoxBase,
    SubtitleLabel,
    LineEdit,
    CaptionLabel,
    HeaderCardWidget,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    IndeterminateProgressRing,
    TeachingTip,
    InfoBarIcon,
    TeachingTipTailPosition,
    ToolTipFilter,
    ToolTipPosition
)
from metabci.brainda.datasets import Wang2016, Nakanishi2015
import resource_rc

from core.plot import MultiSelectComboBox, PlotWidget, DataManager


class SettingMessageBox(MessageBoxBase):
    """
    Custom message box for parameter settings
    
    This dialog allows users to input data analysis parameters including:
    - Delay time (seconds)
    - Sampling rate (Hz)
    - Data duration (seconds)
    - Filter type
    - Channel selection
    """

    def __init__(self, parent=None):
        """
        Initialize the settings dialog
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Filter type selection combobox
        self.filterComboBox = ComboBox(self)
        self.filterComboBox.addItems(["bandpass","lowpass", "highpass", "notch", "butter"])
        
        # Title label
        self.titleLabel = SubtitleLabel("Input Data Parameters", self)
        
        # Delay time input field
        self.delayLineEdit = LineEdit(self)
        self.delayLineEdit.setPlaceholderText("Input delay time (seconds) (default 0.14s)")
        
        # Sampling rate input field
        self.srateLineEdit = LineEdit(self)
        self.srateLineEdit.setPlaceholderText("Input sampling rate (Hz) (default 250Hz)")
        
        # Duration input field
        self.durationLineEdit = LineEdit(self)
        self.durationLineEdit.setPlaceholderText("Input data duration (seconds) (default 5s)")
        
        # Channel selection combobox
        self.channelComboBox = ComboBox(self)
        self.channelComboBox.addItems(["Occipital 9 channels","All channels"])

        # Enable clear button for input fields
        self.delayLineEdit.setClearButtonEnabled(True)
        self.srateLineEdit.setClearButtonEnabled(True)
        self.durationLineEdit.setClearButtonEnabled(True)

        # Warning label for invalid inputs
        self.warningLabel = CaptionLabel("Please enter valid parameter values")
        self.warningLabel.setTextColor("#e40101", QColor(255, 28, 32))

        # Add widgets to view layout
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.delayLineEdit)
        self.viewLayout.addWidget(self.srateLineEdit)
        self.viewLayout.addWidget(self.durationLineEdit)
        self.viewLayout.addWidget(self.filterComboBox)
        self.viewLayout.addWidget(self.channelComboBox)
        self.viewLayout.addWidget(self.warningLabel)

        # Hide warning label initially
        self.warningLabel.hide()

        # Customize button text
        self.yesButton.setText("Confirm")
        self.cancelButton.setText("Cancel")

        # Set minimum width for the dialog
        self.widget.setMinimumWidth(350)

        # self.hideYesButton()

    def getParameters(self):
        """
        Get user input parameters
        
        Returns:
            dict: Dictionary containing all parameters or None if invalid input
        """
        try:
            delay_val = float(self.delayLineEdit.text()) if self.delayLineEdit.text() else float(0.14)
            srate_val = int(self.srateLineEdit.text()) if self.srateLineEdit.text() else float(250)
            duration_val = float(self.durationLineEdit.text()) if self.durationLineEdit.text() else float(5)
            filter_type = self.filterComboBox.currentText()
            channels = self.channelComboBox.currentText()

            return {
                'delay': delay_val,
                'srate': srate_val,
                'duration': duration_val,
                'filter_type': filter_type,
                "channels": channels
            }
        except ValueError:
            return None

    def validate(self):
        """
        Validate input parameters
        
        Returns:
            bool: True if all parameters are valid, False otherwise
        """
        params = self.getParameters()
        if params is None:
            self.warningLabel.setText("Please enter valid numerical parameters")
            self.warningLabel.show()
            return False

        self.warningLabel.hide()
        return True


class SettingCard(GroupHeaderCardWidget):
    """
    Main settings card widget for data analysis
    
    This widget contains all the controls for:
    - Parameter configuration
    - Dataset and subject selection
    - Data loading
    - Frequency selection
    - Plot generation
    """
    
    def __init__(self, parent=None):
        """
        Initialize the settings card
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setTitle("Basic Settings")
        self.setBorderRadius(8)

        # Initialize data manager for handling dataset operations
        self.data_manager = DataManager()
        self.data_manager.dataLoaded.connect(self.on_data_loaded)
        self.data_manager.loadingStarted.connect(self.on_loading_started)
        self.data_manager.loadingFinished.connect(self.on_loading_finished)
        self.data_manager.loadingError.connect(self.on_loading_error)

        # Settings button
        self.chooseButton = PushButton(text="Settings", icon=FluentIcon.SETTING)
        self.chooseButton.setToolTip("Set data analysis parameters")
        self.chooseButton.installEventFilter(
            ToolTipFilter(self.chooseButton, showDelay=300, position=ToolTipPosition.TOP)
        )

        self.chooseButton.clicked.connect(self.chooseSlot)

        # Dataset and subject selection controls
        self.dataContainer = QWidget()
        self.datasetcomboBox = ComboBox()
        self.subjectcomboBox = ComboBox()

        # Progress bar for data loading
        self.progressBar = IndeterminateProgressRing()
        self.progressBar.setFixedSize(20,20)
        self.progressBar.setVisible(False)
        self.progressBar.setValue(0)

        # Layout for data selection controls
        self.datavBoxLayout = QHBoxLayout(self.dataContainer)
        self.datavBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.datavBoxLayout.setSpacing(8)

        # Configure dataset selection combobox
        self.chooseButton.setFixedWidth(120)
        self.datasetcomboBox.setFixedWidth(120)
        self.datasetcomboBox.setToolTip("Select dataset")
        self.datasetcomboBox.setPlaceholderText("Please select a dataset...")
        self.datasetcomboBox.installEventFilter(
            ToolTipFilter(
                self.datasetcomboBox, showDelay=100, position=ToolTipPosition.TOP
            )
        )
        self.datasetcomboBox.addItem("Wang2016", userData=0)
        self.datasetcomboBox.addItem("Nakanishi2015", userData=1)
        self.datasetcomboBox.setCurrentIndex(-1)
        self.datasetcomboBox.setMinimumWidth(160)
        self.datasetcomboBox.currentIndexChanged.connect(self.datacomboSlot)

        # Configure subject selection combobox
        self.subjectcomboBox.setMinimumWidth(160)
        self.subjectcomboBox.setPlaceholderText("Please select a subject...")
        self.subjectcomboBox.currentIndexChanged.connect(self.subjectcomboSlot)
        self.subjectcomboBox.setCurrentIndex(-1)
        self.subjectcomboBox.setEnabled(False)  # Initially disabled

        # First row: dataset and subject selection
        firstRowContainer = QWidget()
        firstRowLayout = QHBoxLayout(firstRowContainer)
        firstRowLayout.setContentsMargins(0, 0, 0, 0)
        firstRowLayout.setSpacing(8)
        firstRowLayout.addWidget(self.datasetcomboBox)
        firstRowLayout.addWidget(self.subjectcomboBox)

        # Add to container
        self.datavBoxLayout.addWidget(firstRowContainer)
        self.datavBoxLayout.addWidget(self.progressBar)

        # Plot container - add plot mode selection
        self.plotContainer = QWidget()
        self.plotvBoxLayout = QHBoxLayout(self.plotContainer)
        self.plotvBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.plotvBoxLayout.setSpacing(8)

        # Frequency selection
        self.freqComboBox = MultiSelectComboBox(self)     
        self.freqComboBox.setText("Please select frequencies...")   
        self.freqComboBox.selectionChanged.connect(self.freqcomboSlot)
        self.freqComboBox.setEnabled(False)  # Initially disabled

        # Plot mode selection
        self.plotModeComboBox = ComboBox(self)
        self.plotModeComboBox.addItems(["Single overlay", "Separate subplots"])
        self.plotModeComboBox.setToolTip("Select plot mode")
        self.plotModeComboBox.installEventFilter(
            ToolTipFilter(
                self.plotModeComboBox, showDelay=100, position=ToolTipPosition.TOP
            )
        )
        self.plotModeComboBox.setCurrentIndex(0)  # Default to single overlay
        self.plotModeComboBox.setFixedWidth(100)

        # Plot button
        self.plotButton = PushButton("Generate Plot")
        self.plotButton.clicked.connect(self.plotButtonClicked)
        self.plotButton.setEnabled(False)

        # Add plot controls to layout
        self.plotvBoxLayout.addWidget(self.freqComboBox)
        self.plotvBoxLayout.addWidget(self.plotModeComboBox)
        self.plotvBoxLayout.addWidget(self.plotButton)

        # Store current parameters and state
        self.current_params = {}
        self.selected_frequencies = []
        self.current_dataset = None
        self.current_subject = None
        self.current_data = None

        # Add groups to the card
        self.addGroup(
            FluentIcon.SETTING,
            "Parameter Settings",
            "Set analysis parameters for the dataset",
            self.chooseButton,
        )

        self.addGroup(
            FluentIcon.UPDATE, "Load Dataset", "Load the dataset to be analyzed", self.dataContainer
        )
        self.addGroup(
            FluentIcon.PALETTE,
            "Generate Plot",
            "Preprocess data and generate frequency domain plots",
            self.plotContainer,
        )

    def datacomboSlot(self):
        """
        Handle dataset selection change
        Updates the subject selection based on the chosen dataset
        """
        print("Dataset selection changed")

        dataset_name:str = self.datasetcomboBox.currentText()
        self.subjectcomboBox.blockSignals(True)
        if dataset_name != "":
            self.current_dataset = dataset_name

            # Update subject selection
            self.subjectcomboBox.clear()
            self.subjectcomboBox.setEnabled(True)

            # Set maximum subjects based on dataset
            if dataset_name == "Wang2016":
                max_subjects = 35
            elif dataset_name == "Nakanishi2015":
                max_subjects = 10
            else:
                max_subjects = 1

            # Add subject options
            for i in range(1, max_subjects + 1):
                self.subjectcomboBox.addItem(f"Subject {i}", userData=i)

            self.subjectcomboBox.setCurrentIndex(-1)

            # Set dataset in data manager
            self.data_manager.set_dataset(dataset_name)
            self.subjectcomboBox.blockSignals(False)
            
            # Clear frequency selection and disable plotting
            self.freqComboBox.clear_selection()
            self.plotButton.setEnabled(False)
            self.plotButton.setEnabled(False)
            
            # Update table display
            self.updateTableDataset(dataset_name=dataset_name)

    def _load_dataset_info(self, dataset_name):
        """
        Load dataset information including events, frequencies, and channels
        
        Args:
            dataset_name (str): Name of the dataset to load
        """
        try:
            if dataset_name == "Wang2016":
                dataset = Wang2016()
            elif dataset_name == "Nakanishi2015":
                dataset = Nakanishi2015()
            else:
                return

            # Get basic dataset information
            events = sorted(list(dataset.events.keys()))
            freqs = [dataset.get_freq(event) for event in events]
            channels = dataset._CHANNELS

            self.dataset_info = {
                'events': events,
                'freqs': freqs,
                'channels': channels,
                'dataset': dataset
            }

            print(f"Dataset {dataset_name} information:")
            print(f"  Frequencies: {freqs}")
            print(f"  Channel count: {len(channels)}")
            print(f"  Channels: {channels}")

        except Exception as e:
            print(f"Failed to load dataset information: {e}")
            self.showInfoBar(f"Failed to load dataset information: {e}", "error")

    def _get_actual_channels(self, selected_channel_type, dataset_name):
        """
        Get actual channel list based on selected channel type and dataset
        
        Args:
            selected_channel_type (str): Type of channels selected by user
            dataset_name (str): Name of the dataset
            
        Returns:
            list: List of actual channel names
        """
        if not self.dataset_info:
            return []

        available_channels = self.dataset_info['channels']

        if dataset_name == "Nakanishi2015":
            # Nakanishi2015 dataset only has 8 channels, use all channels regardless of selection
            return available_channels

        if selected_channel_type == "Occipital 9 channels":
            # Standard occipital 9-channel setup
            occipital_channels = ["PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]
            # Only return channels that exist in the dataset
            return [ch for ch in occipital_channels if ch in available_channels]
        elif selected_channel_type == "All channels":
            return available_channels
        else:
            return available_channels

    def subjectcomboSlot(self):
        """
        Handle subject selection change
        Initiates data loading for the selected subject
        """
        print("Subject selection changed")
        if self.subjectcomboBox.currentIndex() >= 0:
            subject_id = self.subjectcomboBox.currentData()
            if subject_id and self.current_dataset:
                self.current_subject = subject_id

                # Start loading data
                if not self.current_params:
                    self.showInfoBar("Please set parameters before loading data!", "warning")
                    return

                self._load_dataset_info(self.current_dataset)
                selected_channels = self.current_params.get("channels", "Occipital 9 channels")
                actual_channels = self._get_actual_channels(selected_channels, self.current_dataset)
                if not actual_channels:
                    self.showInfoBar("Unable to determine channel list", "error")
                    return

                self.data_manager.load_data(subject_id=subject_id, 
                                            srate=self.current_params.get("srate",250),
                                            selected_channels=actual_channels,
                                            duration=self.current_params.get("duration", 5),
                                            delay=self.current_params.get("delay", 0.14))

                # Update table display
                self.updateTableDataset(subject_id=f"Subject {subject_id}")

    def on_loading_started(self):
        """
        Handle data loading start
        Shows progress bar and disables controls
        """
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.subjectcomboBox.setEnabled(False)
        self.datasetcomboBox.setEnabled(False)
        self.plotButton.setEnabled(False)

        # Connect progress signal
        progress_signal = self.data_manager.get_progress_signal()
        if progress_signal:
            progress_signal.connect(self.progressBar.setValue)

        self.showInfoBar("Loading dataset...", "info")

    def updateTableFrequencyRange(self, freqs):
        """
        Update frequency range information in the table
        
        Args:
            freqs (list): List of available frequencies
        """
        if freqs:
            freq_range_text = f"{min(freqs)}-{max(freqs)} Hz"
            analyze_widget = self.window().findChild(AnalyzeWidget)
            if analyze_widget and analyze_widget.dataTable:
                analyze_widget.dataTable.dataTale.tableView.setItem(
                    7, 1, QTableWidgetItem(freq_range_text)
                )
                analyze_widget.dataTable.dataTale.tableView.setItem(
                    7, 2, QTableWidgetItem("Provided by dataset")
                )
                analyze_widget.dataTable.dataTale.tableView.setItem(
                    7, 3, QTableWidgetItem(f"Total {len(freqs)} frequencies available")
                )

    def on_data_loaded(self, data_info):
        """
        Handle successful data loading
        Updates frequency selection and enables plot button
        
        Args:
            data_info (dict): Information about the loaded data
        """
        self.current_data = data_info

        # Update frequency selection
        freqs = data_info['freqs']
        self.freqComboBox.clear_selection()
        self.freqComboBox.all_items.clear()
        self.freqComboBox.checkboxes.clear()
        self.freqComboBox.original_texts.clear()
        self.freqComboBox.menu.clear()

        for freq in freqs:
            tooltip = f"Frequency: {freq} Hz"
            self.freqComboBox.addItem(str(freq), tooltip=tooltip)

        # Enable plot button
        self.freqComboBox.setEnabled(True)
        self.plotButton.setEnabled(True)

        self.updateTableFrequencyRange(freqs)
        self.showInfoBar("Data loading completed!", "success")

    def on_loading_finished(self):
        """
        Handle data loading completion
        Re-enables controls and hides progress bar
        """
        self.progressBar.setVisible(False)
        self.subjectcomboBox.setEnabled(True)
        self.datasetcomboBox.setEnabled(True)

    def on_loading_error(self, error_msg):
        """
        Handle data loading error
        
        Args:
            error_msg (str): Error message to display
        """
        self.progressBar.setVisible(False)
        self.subjectcomboBox.setEnabled(True)
        self.datasetcomboBox.setEnabled(True)
        self.showInfoBar(f"Data loading failed: {error_msg}", "error")

    def freqcomboSlot(self, selected_freqs):
        """
        Handle frequency selection change
        Updates the frequency information in the table
        
        Args:
            selected_freqs (list): List of selected frequencies
        """
        # Get data table from parent window
        analyze_widget = self.window().findChild(AnalyzeWidget)
        if analyze_widget and analyze_widget.dataTable:
            self.selected_frequencies = selected_freqs
            analyze_widget.dataTable.dataTale.updateSelectedFrequencies(selected_freqs)

    def chooseSlot(self):
        """
        Open parameter settings dialog
        Shows teaching tip for Nakanishi2015 dataset
        """
        TeachingTip.create(
            target=self.chooseButton,
            icon=InfoBarIcon.INFORMATION,
            title="Tips!",
            content="For Nakanishi2015 dataset, please set sampling rate to 256Hz, maximum trial length 4s",
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2000,
            parent=self,
        )
        mainwindow = self.window()
        w = SettingMessageBox(mainwindow)
        if w.exec():
            parms = w.getParameters()
            if parms:
                self.current_params.update(parms)
                self.updateTableParams(parms)

                # If subject is already selected, prompt to reload data
                if self.current_subject:
                    self.showInfoBar("Parameters updated, please reselect subject to load data", "info")
                    # Reset subject selection
                    self.subjectcomboBox.setCurrentIndex(-1)
                    self.current_data = None
                    self.freqComboBox.setEnabled(False)
                    self.plotButton.setEnabled(False)

    def updateTableDataset(self, dataset_name=None, subject_id=None):
        """
        Update dataset and subject ID in the data table
        
        Args:
            dataset_name (str, optional): Name of the selected dataset
            subject_id (str, optional): ID of the selected subject
        """
        analyze_widget = self.window().findChild(AnalyzeWidget)
        if analyze_widget and analyze_widget.dataTable:
            analyze_widget.dataTable.dataTale.updateDatasetInfo(dataset_name, subject_id)

    def updateTableParams(self, params):
        """
        Update parameter information in the table
        
        Args:
            params (dict): Dictionary of parameters to update
        """
        # Get data table from parent window
        analyze_widget = self.window().findChild(AnalyzeWidget)
        if analyze_widget and analyze_widget.dataTable:
            analyze_widget.dataTable.dataTale.updateParams(params)

    def plotButtonClicked(self):
        """
        Handle plot button click event
        Validates inputs and initiates plot generation
        """
        try:
            if not self.current_data:
                self.showInfoBar("Please load data first!", "warning")
                return
                
            # Check if required parameters are set
            if not self.selected_frequencies:
                self.showInfoBar("Please select frequencies to analyze first!", "warning")
                return

            if self.current_dataset is None:
                self.showInfoBar("Please select a dataset first!", "warning")
                return

            if self.current_subject is None:
                self.showInfoBar("Please select a subject first!", "warning")
                return

            # Get plot mode
            plot_mode_text = self.plotModeComboBox.currentText()
            plot_mode = "single" if plot_mode_text == "Single overlay" else "subplot"

            # Get parameters, use defaults if not set
            analysis_params = {
                "srate": self.current_params.get("srate", 250),
                "delay": self.current_params.get("delay", 0.14),
                "duration": self.current_params.get("duration", 5),
                "filter_type": self.current_params.get("filter_type", "bandpass"),
                "channels": self.current_params.get("channels", "Occipital 9 channels"),
            }

            # Show processing information
            self.showInfoBar("Processing data and generating plots...", "info")

            # Get plot widget
            analyze_widget = self.window().findChild(AnalyzeWidget)
            if analyze_widget and analyze_widget.plotWidget:
                # Execute plotting
                success = analyze_widget.plotWidget.plotFFTAnalysis(
                    self.current_data,
                    analysis_params,
                    self.selected_frequencies,
                    analysis_params["channels"],
                    plot_mode
                )

                if success:
                    self.showInfoBar("Plot generation completed!", "success")
                else:
                    self.showInfoBar("Plot generation failed!", "error")
            else:
                self.showInfoBar("Plot widget not found!", "error")

        except Exception as e:
            self.showInfoBar(f"Error occurred during plotting: {str(e)}", "error")
            print(f"Plotting error: {e}")

    def showInfoBar(self, message, type="info"):
        """
        Display information bar with message
        
        Args:
            message (str): Message to display
            type (str): Type of message ("success", "warning", "error", "info")
        """
        if type == "success":
            InfoBar.success(
                title="Success",
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self.window(),
            )
        elif type == "warning":
            InfoBar.warning(
                title="Warning",
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self.window(),
            )
        elif type == "error":
            InfoBar.error(
                title="Error",
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=4000,
                parent=self.window(),
            )
        else:  # info
            InfoBar.info(
                title="Information",
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self.window(),
            )


class DataTable(QWidget):
    """
    Data table widget for displaying analysis parameters and status
    
    This widget shows:
    - Dataset information
    - Subject ID
    - Analysis parameters (delay, sampling rate, duration, etc.)
    - Filter settings
    - Channel configuration
    - Frequency ranges
    """

    def __init__(self, parent=None):
        """
        Initialize the data table
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.hBoxLayout = QHBoxLayout(self)
        self.tableView = TableWidget(self)
        self.tableView.setBorderVisible(True)
        self.tableView.setBorderRadius(8)

        self.tableView.setWordWrap(False)
        self.tableView.setRowCount(8)
        self.tableView.setColumnCount(4)

        self.tableView.verticalHeader().hide()
        self.tableView.setHorizontalHeaderLabels(
            ["Parameter Name", "Parameter Value", "Status", "Notes"]
        )

        self.initializeTable()
        # self.tableView.resizeColumnsToContents()
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableView.setSortingEnabled(True)

        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.hBoxLayout.addWidget(self.tableView)

    def initializeTable(self):
        """
        Initialize table with default parameter values
        """
        default_params = [
            ("Dataset", "Not selected", "Not set", "Please select a dataset first"),
            ("Subject ID", "Not selected", "Not set", "Please select a subject first"),
            ("Delay time", f"{0.14} seconds", "Default value", "Can be modified through parameter settings"),
            ("Sampling rate", f"{250} Hz", "Default value", "Can be modified through parameter settings"),
            ("Duration", f"{5} seconds", "Default value", "Can be modified through parameter settings"),
            ("Filter", "Not set", "Not set", "Can be modified through parameter settings"),
            ("Channel selection", "Not set", "Not set", "Can be modified through parameter settings"),
            ("Frequency range", "To be displayed after data loading", "Not set", "Please load data first"),
        ]

        for i, (param_name, param_value, status, note) in enumerate(default_params):
            self.tableView.setItem(i, 0, QTableWidgetItem(param_name))
            self.tableView.setItem(i, 1, QTableWidgetItem(param_value))
            self.tableView.setItem(i, 2, QTableWidgetItem(status))
            self.tableView.setItem(i, 3, QTableWidgetItem(note))

    def updateDatasetInfo(self, dataset_name=None, subject_id=None):
        """
        Update dataset and subject information
        
        Args:
            dataset_name (str, optional): Name of the selected dataset
            subject_id (str, optional): ID of the selected subject
        """
        # Update dataset
        if dataset_name:
            self.tableView.setItem(0, 1, QTableWidgetItem(dataset_name))
            self.tableView.setItem(0, 2, QTableWidgetItem("Set"))
            self.tableView.setItem(0, 3, QTableWidgetItem("User selection"))

        # Update subject ID
        if subject_id:
            self.tableView.setItem(1, 1, QTableWidgetItem(subject_id))
            self.tableView.setItem(1, 2, QTableWidgetItem("Set"))
            self.tableView.setItem(1, 3, QTableWidgetItem("User selection"))

    def updateSelectedFrequencies(self, selected_freqs):
        """
        Update selected frequency information
        
        Args:
            selected_freqs (list): List of selected frequencies
        """
        if not selected_freqs:
            # No frequencies selected
            self.tableView.setItem(
                7, 1, QTableWidgetItem(f"{8}-{15} Hz")
            )
            self.tableView.setItem(7, 2, QTableWidgetItem("Not set"))
            self.tableView.setItem(7, 3, QTableWidgetItem("Please select frequencies to analyze"))
        else:
            # Format frequency display
            freq_text = ", ".join(selected_freqs)
            if len(freq_text) > 30:  # Truncate if too long
                freq_text = freq_text[:27] + "..."

            # Add Hz unit
            display_text = f"{freq_text} Hz"

            self.tableView.setItem(7, 1, QTableWidgetItem(display_text))
            self.tableView.setItem(7, 2, QTableWidgetItem("Set"))
            self.tableView.setItem(
                7, 3, QTableWidgetItem(f"Total {len(selected_freqs)} frequencies selected")
            )

    def updateParams(self, params):
        """
        Update parameter information in the table
        
        Args:
            params (dict): Dictionary of parameters to update
        """
        if not params:
            return

        # Mapping of parameter names to row numbers
        param_row_map = {
            "delay": 2,
            "srate": 3,
            "duration": 4,
            "filter_type": 5,
            "channels": 6,
        }

        for param_name, value in params.items():
            if param_name in param_row_map:
                row = param_row_map[param_name]

                # Format display value
                display_value = self.formatParamValue(param_name, value)

                self.tableView.setItem(row, 1, QTableWidgetItem(display_value))
                self.tableView.setItem(row, 2, QTableWidgetItem("Set"))
                self.tableView.setItem(row, 3, QTableWidgetItem("User defined"))

    def formatParamValue(self, param_name, value):
        """
        Format parameter value for display
        
        Args:
            param_name (str): Name of the parameter
            value: Value of the parameter
            
        Returns:
            str: Formatted value string
        """
        if param_name == "delay":
            return f"{value} seconds"
        elif param_name == "srate":
            return f"{value} Hz"
        elif param_name == "duration":
            return f"{value} seconds"
        elif param_name == "filter_type":
            return str(value)
        elif param_name == "channels":
            return str(value)
        else:
            return str(value)


class DataTableCard(HeaderCardWidget):
    """
    Header card widget containing the data table
    """
    
    def __init__(self, parent=None):
        """
        Initialize the data table card
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setTitle("Data Table")
        # self.vBoxLayout = QVBoxLayout(self)
        self.dataTale = DataTable(self)
        self.viewLayout.addWidget(self.dataTale)


class AnalyzeWidget(ScrollArea):
    """
    Main analysis widget containing all analysis components
    
    This widget provides:
    - Settings card for parameter configuration
    - Data table for parameter display
    - Plot widget for visualization
    """
    
    def __init__(self, parent=None):
        """
        Initialize the analyze widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        # self.setWindowTitle("Analyze")

        # Create central widget
        self.view = QWidget(self)

        # Create components
        self.settingsCard = SettingCard(self.view)
        self.dataTable = DataTableCard(self.view)
        self.plotWidget = PlotWidget(self.view)

        self.setupLayout()
        self.settingsCard.setFixedHeight(200)
        
        # Set transparent background
        self.view.setAutoFillBackground(False)  # Don't use setStyleSheet
        palette = self.view.palette()
        palette.setColor(self.view.backgroundRole(), Qt.transparent)
        self.view.setPalette(palette)

        # Set ScrollArea's central widget
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea{border: none; background: transparent}")


    def setupLayout(self):
        """
        Set up the layout for all components
        """
        # Create main layout
        Layout = QVBoxLayout(self.view)

        # self.setLayout(mainLayout)
        # Set layout properties
        Layout.setContentsMargins(20, 20, 20, 20)
        Layout.setSpacing(20)
        
        # Add settings card to main layout
        Layout.addWidget(self.settingsCard, 0, Qt.AlignTop)

        # Add data table to main layout
        Layout.addWidget(self.dataTable, 1, Qt.AlignTop)
        
        # Add plot widget to main layout
        Layout.addWidget(self.plotWidget, 0, Qt.AlignTop)

        # Ensure content is tall enough to show scroll bar
        self.settingsCard.setFixedHeight(220)
        self.plotWidget.setMinimumHeight(700)
        self.dataTable.setMinimumHeight(400)
        

        # self.setLayout(Layout)
