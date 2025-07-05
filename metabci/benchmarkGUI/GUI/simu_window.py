"""
Simulation Window Module

This module provides the simulation interface for EEG data analysis.
It includes components for dataset selection, simulation control, and real-time output display.

Features:
- Dataset folder selection
- Background simulation execution with threading
- Real-time output display with queue-based communication
- Start/stop simulation controls
- Progress monitoring and error handling

Author: DSG
Date: 2025-07-03
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon, QTextCursor
from qfluentwidgets import FluentIcon, ScrollArea, PushButton, GroupHeaderCardWidget
import sys
import queue
import threading
from pathlib import Path


class SimulationThread(QThread):
    """
    Simulation analysis thread - uses queue to pass output
    
    This thread runs the simulation analysis in the background and communicates
    with the main thread through a queue to provide real-time output updates.
    
    Signals:
        finished_signal: Emitted when simulation completes successfully
        error_signal: Emitted when an error occurs during simulation
    """
    finished_signal = Signal()
    error_signal = Signal(str)
    
    def __init__(self, folder_path, output_queue):
        """
        Initialize the simulation thread
        
        Args:
            folder_path (str): Path to the dataset folder
            output_queue (queue.Queue): Queue for passing output messages
        """
        super().__init__()
        self.folder_path = folder_path
        self.output_queue = output_queue
        self.stop_event = None
        
    def stop(self):
        """
        Stop the simulation thread gracefully
        """
        if self.stop_event:
            self.stop_event.set()
        
    def run(self):
        """
        Run simulation analysis in background thread
        
        This method:
        1. Imports the simulation algorithm module
        2. Executes the simulation with the selected dataset
        3. Handles errors and provides status updates
        4. Cleans up resources when finished
        """
        try:
            # Create stop event for graceful shutdown
            self.stop_event = threading.Event()
            
            self.output_queue.put("Starting to import simulation algorithm modules...")
            
            # Add simu_algorithm directory to Python path
            simu_algorithm_path = Path(__file__).parent / "simu_algorithm"
            if str(simu_algorithm_path) not in sys.path:
                sys.path.insert(0, str(simu_algorithm_path))
            
            self.output_queue.put("Importing required modules...")
            
            # Dynamically import simulation algorithm module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "simu_datasets", 
                simu_algorithm_path / "Simu_Ds.py"
            )
            simu_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(simu_module)
            
            self.output_queue.put("Module import completed")
            self.output_queue.put(f"Starting analysis of folder: {self.folder_path}")
            
            # Call simu_run function with output queue
            try:
                simu_module.simu_run(
                    self.folder_path, 
                    stop_event=self.stop_event, 
                    max_duration=300,
                    output_queue=self.output_queue
                )
            except Exception as simu_error:
                # Handle socket errors specifically (common when stopping simulation)
                if "WinError 10038" in str(simu_error) or "非套接字" in str(simu_error):
                    self.output_queue.put("Socket error detected, this usually occurs when stopping simulation and can be ignored")
                else:
                    raise simu_error
            
            if not self.stop_event.is_set():
                self.output_queue.put("Simulation analysis completed!")
                self.finished_signal.emit()
            else:
                self.output_queue.put("Simulation has been stopped")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error_signal.emit(f"Error: {str(e)}\nDetails:\n{error_details}")
        finally:
            # Clean up added path
            simu_algorithm_path = Path(__file__).parent / "simu_algorithm"
            if str(simu_algorithm_path) in sys.path:
                sys.path.remove(str(simu_algorithm_path))


class FolderSelectCard(GroupHeaderCardWidget):
    """
    Folder selection card widget
    
    This widget allows users to select a dataset folder containing
    training data for simulation analysis.
    """

    def __init__(self, parent=None):
        """
        Initialize the folder selection card
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setTitle("Dataset Selection")
        self.selected_folder = None

        # Create folder selection button
        self.selectButton = PushButton("Select Dataset Folder", icon=FluentIcon.FOLDER_ADD)
        self.selectButton.clicked.connect(self.select_folder)
        self.selectButton.setFixedWidth(180)  # Set fixed width

        # Label to display selected path
        self.pathLabel = QLabel("No folder selected")
        self.pathLabel.setWordWrap(True)
        
        layout = QHBoxLayout()
        layout.addWidget(self.selectButton)
        layout.addWidget(self.pathLabel)
        # layout.addSpacerItem(QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        container = QWidget()
        container.setLayout(layout)

        # Add components to card
        self.addGroup(
            FluentIcon.BOOK_SHELF,
            "Dataset Selection",
            "Select folder containing training data",
            container,
        )

    def select_folder(self):
        """
        Open folder selection dialog
        """
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if folder_path:
            self.selected_folder = folder_path
            self.pathLabel.setText(f"Selected: {folder_path}")

    def get_selected_folder(self):
        """
        Get the selected folder path
        
        Returns:
            str: Path to the selected folder, or None if no folder selected
        """
        return self.selected_folder


class SimulationControlCard(GroupHeaderCardWidget):
    """
    Simulation control card widget
    
    This widget provides controls for starting and stopping
    the simulation analysis process.
    """

    def __init__(self, parent=None):
        """
        Initialize the simulation control card
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setTitle("Simulation Control")
        
        # Create start simulation button
        self.startButton = PushButton("Start Simulation", icon=FluentIcon.PLAY)
        
        # Create stop simulation button
        self.stopButton = PushButton("Stop Simulation", icon=FluentIcon.PAUSE)
        self.stopButton.setEnabled(False)
        
        # Container for control buttons
        self.simu_container = QWidget()
        self.simu_layout = QHBoxLayout(self.simu_container)
        self.simu_layout.addWidget(self.startButton)
        self.simu_layout.addWidget(self.stopButton)
        
        # Add components to card
        self.addGroup(
            FluentIcon.SPEED_HIGH,
            "Simulation Control",
            "Start or stop simulation analysis",
            self.simu_container,
        )


class SimuWidget(ScrollArea):
    """
    Main simulation widget containing all simulation components
    
    This widget provides:
    - Dataset folder selection
    - Simulation control (start/stop)
    - Real-time output display
    - Background thread management
    """
    
    def __init__(self, parent=None):
        """
        Initialize the simulation widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create central widget
        self.view = QWidget(self)
        
        # Create components
        self.folderSelectCard = FolderSelectCard(self.view)
        self.simulationControlCard = SimulationControlCard(self.view)
        
        # Create output display area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(600)
        self.output_text.setPlaceholderText("Simulation output will be displayed here...")
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        
        # Create output queue for thread communication
        self.output_queue = queue.Queue()
        
        # Create timer to periodically check queue
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_output_from_queue)
        self.update_timer.setInterval(100)  # Check every 100ms
        
        # Simulation thread reference
        self.simulation_thread = None
        
        # Connect signals
        self.simulationControlCard.startButton.clicked.connect(self.start_simulation)
        self.simulationControlCard.stopButton.clicked.connect(self.stop_simulation)
        
        self.setupLayout()
        
        # Set transparent background
        self.view.setAutoFillBackground(False)
        palette = self.view.palette()
        palette.setColor(self.view.backgroundRole(), Qt.transparent)
        self.view.setPalette(palette)

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
        
        # Set layout properties
        Layout.setContentsMargins(20, 20, 20, 20)
        Layout.setSpacing(20)
        
        # Add cards to main layout
        # Layout.addWidget(self.folderSelectCard, 0, Qt.AlignTop)
        # Layout.addWidget(self.simulationControlCard, 0, Qt.AlignTop)
        Layout.addWidget(self.folderSelectCard)
        Layout.addWidget(self.simulationControlCard)
        
        # Add output display area
        Layout.addWidget(self.output_text)
        
        # Add spacer to push content to top
        from PySide6.QtWidgets import QSpacerItem, QSizePolicy
        Layout.addItem(QSpacerItem(20, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Set minimum heights for cards
        self.folderSelectCard.setMinimumHeight(140)
        self.simulationControlCard.setMinimumHeight(140)
    
    def start_simulation(self):
        """
        Start simulation analysis
        
        This method:
        1. Validates that a dataset folder is selected
        2. Clears previous output
        3. Creates and starts the simulation thread
        4. Updates button states
        """
        # Check if folder is selected
        folder_path = self.folderSelectCard.get_selected_folder()
        if not folder_path:
            self.append_output("Please select a dataset folder first!")
            return
        
        # Check if simulation is already running
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.append_output("Simulation is already running, please wait for completion...")
            return
        
        # Clear output area and queue
        self.output_text.clear()
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        
        # Update button states
        self.simulationControlCard.startButton.setEnabled(False)
        self.simulationControlCard.stopButton.setEnabled(True)
        
        # Start timer to check output queue
        self.update_timer.start()
        
        # Create and start simulation thread
        self.simulation_thread = SimulationThread(folder_path, self.output_queue)
        self.simulation_thread.finished_signal.connect(self.simulation_finished)
        self.simulation_thread.error_signal.connect(self.simulation_error)
        self.simulation_thread.start()
        
        self.append_output(f"Starting simulation analysis, dataset path: {folder_path}")
    
    def stop_simulation(self):
        """
        Stop simulation analysis
        
        This method attempts to gracefully stop the simulation thread
        and falls back to forced termination if necessary.
        """
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.append_output("Stopping simulation...")
            
            # First set the stop event
            self.simulation_thread.stop()
            
            # Give thread some time to exit gracefully
            self.simulation_thread.wait(8000)  # Wait up to 8 seconds
            
            # If still running, force terminate
            if self.simulation_thread.isRunning():
                self.append_output("Force terminating simulation thread...")
                self.simulation_thread.terminate()
                self.simulation_thread.wait(1000)  # Wait another 1 second
            
            self.append_output("Simulation stopped")
        
        self.simulation_finished()
    
    def simulation_finished(self):
        """
        Handle simulation completion
        
        Re-enables controls and processes any remaining output
        """
        self.simulationControlCard.startButton.setEnabled(True)
        self.simulationControlCard.stopButton.setEnabled(False)
        self.update_timer.stop()  # Stop timer
        
        # Process remaining output in queue
        self.update_output_from_queue()
        
        self.append_output("=" * 50)
        self.append_output("Simulation analysis completed!")
    
    def simulation_error(self, error_msg):
        """
        Handle simulation error
        
        Args:
            error_msg (str): Error message from the simulation thread
        """
        self.append_output(f"Simulation error: {error_msg}")
        self.simulation_finished()
    
    def update_output_from_queue(self):
        """
        Update output from queue to interface
        
        This method runs periodically to check for new output messages
        from the simulation thread and display them in the output area.
        """   
        try:
            while True:
                try:
                    output = self.output_queue.get_nowait()
                    self.append_output(output)
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error updating output: {e}")
    
    def append_output(self, text):
        """
        Append output text to display area
        
        This method handles text formatting and automatic scrolling
        to ensure the output area stays up-to-date with the latest messages.
        
        Args:
            text (str): Text to append to the output display
        """
        # Handle empty text and newlines
        if not text:
            return
            
        # Remove trailing newlines but preserve content
        text = text.rstrip('\n\r')
        
        # If it's an empty line, add an empty line
        if not text.strip():
            self.output_text.append("")
        else:
            self.output_text.append(text)
            
        # Auto-scroll to bottom
        cursor = self.output_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_text.setTextCursor(cursor)
        
        # Force refresh display
        self.output_text.repaint()
