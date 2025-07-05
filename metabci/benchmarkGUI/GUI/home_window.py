"""
Home Window Module

This module provides the main home interface for the MetaBCI GUI application.
It includes components for application information, feature showcase, function cards,
system requirements, and interactive gallery display.

Features:
- Application information card with project details
- Interactive gallery with lightbox display
- Function cards for external links and demos
- System requirements display
- Feature showcase with flip view

Classes:
- StatisticsWidget: Display numerical statistics with labels
- AppInfoCard: Main application information display
- GalleryCard: Image gallery with flip view
- FunctionCard: Clickable feature cards
- SystemRequirementCard: System compatibility information
- SettinsCard: Basic settings configuration (currently unused)
- LightBox: Full-screen image viewer with animations
- HomeWidget: Main container for all home interface components

Author: DSG
Date: 2025-07-03
"""

# Core Qt imports for UI framework
from PySide6.QtCore import Qt, QPoint, QSize, QUrl, QRect, QPropertyAnimation
from PySide6.QtGui import QFont, QColor, QPainter

# Qt Widgets for UI components
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QGraphicsOpacityEffect
)

# Fluent UI widgets for modern interface design
from qfluentwidgets import (
    ElevatedCardWidget,
    IconWidget,
    BodyLabel,
    CaptionLabel,
    PushButton,
    TransparentToolButton,
    FluentIcon,
    ImageLabel,
    isDarkTheme,
    SimpleCardWidget,
    HeaderCardWidget,
    InfoBarIcon,
    HyperlinkLabel,
    HorizontalFlipView,
    PrimaryPushButton,
    TitleLabel,
    PillPushButton,
    setFont,
    ScrollArea,
    GroupHeaderCardWidget,
    ComboBox,
    SearchLineEdit,
    TeachingTip,
    TeachingTipTailPosition,
    InfoBar,
    InfoBarPosition,
    ToolTipFilter,
    ToolTipPosition,
)

# Additional fluent UI components
from qfluentwidgets.components.widgets.acrylic_label import AcrylicBrush

import resource_rc  
import webbrowser  
import subprocess  
import sys  
from pathlib import Path  

class StatisticsWidget(QWidget):
    """
    Statistics widget for displaying numerical data with title
    
    This widget displays a value with its corresponding title,
    commonly used for showing application statistics or metrics.
    """

    def __init__(self, title: str, value: str, parent=None):
        """
        Initialize the statistics widget
        
        Args:
            title (str): Title text to display
            value (str): Value text to display
            parent: Parent widget
        """
        super().__init__(parent=parent)
        self.titleLabel = CaptionLabel(title, self)
        self.valueLabel = BodyLabel(value, self)
        self.vBoxLayout = QVBoxLayout(self)

        self.vBoxLayout.setContentsMargins(16, 0, 16, 0)
        self.vBoxLayout.addWidget(self.valueLabel, 0, Qt.AlignTop)
        self.vBoxLayout.addWidget(self.titleLabel, 0, Qt.AlignBottom)

        setFont(self.valueLabel, 18, QFont.DemiBold)
        self.titleLabel.setTextColor(QColor(96, 96, 96), QColor(206, 206, 206))


class AppInfoCard(SimpleCardWidget):
    """
    Application information card widget
    
    This card displays comprehensive information about the MetaBCI application
    including icon, name, description, company info, and action buttons.
    """

    def __init__(self, parent=None):
        """
        Initialize the application information card
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.projectUrl = "https://github.com/dsgguo/MetaBenchmark-GUI"
        
        # Application icon
        self.iconLabel = ImageLabel(":/icon/MetaBCI", self)
        self.iconLabel.setBorderRadius(8, 8, 8, 8)
        self.iconLabel.scaledToWidth(120)

        # Application name and buttons
        self.nameLabel = TitleLabel("MetaBCI", self)
        self.openButton = PrimaryPushButton("Open", self)
        self.companyLabel = HyperlinkLabel(
            QUrl("https://www.tju.edu.cn/"), "Tianjin University", self
        )
        self.openButton.setFixedWidth(160)
        self.openButton.clicked.connect(self.openSlot)

        # Application description
        self.descriptionLabel = BodyLabel(
            "This branch of MetaBCI is developed and maintained by the Skynexus team at Tianjin University.",
            self,
        )
        self.descriptionLabel.setWordWrap(True)

        # Share button
        self.shareButton = TransparentToolButton(FluentIcon.SHARE, self)
        self.shareButton.setFixedSize(32, 32)
        self.shareButton.setIconSize(QSize(14, 14))
        self.shareButton.clicked.connect(self.shareSlot)

        # Layout setup - create layout hierarchy for organizing UI components
        self.hBoxLayout = QHBoxLayout(self)  
        self.vBoxLayout = QVBoxLayout()  
        self.topLayout = QHBoxLayout()  
        self.statisticsLayout = QHBoxLayout() 
        self.buttonLayout = QHBoxLayout()  

        self.initLayout()  # Initialize and configure all layouts
        self.setBorderRadius(8)  # Set rounded corners for the card

    def initLayout(self):
        """
        Initialize the layout for the application information card
        
        This method sets up the complete layout hierarchy including:
        - Icon positioning on the left
        - Title and action buttons at the top
        - Company information below title
        - Description text in the middle
        - Action buttons at the bottom
        """
        self.hBoxLayout.setSpacing(30)  
        self.hBoxLayout.setContentsMargins(34, 24, 24, 24)  
        self.hBoxLayout.addWidget(self.iconLabel)  
        self.hBoxLayout.addLayout(self.vBoxLayout)  
        
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)  
        self.vBoxLayout.setSpacing(0)  

        # Name label and open button - top row layout
        self.vBoxLayout.addLayout(self.topLayout)
        self.topLayout.setContentsMargins(0, 0, 0, 0)
        self.topLayout.addWidget(self.nameLabel)  
        self.topLayout.addWidget(self.openButton, 0, Qt.AlignRight)  

        # Company label - positioned below title
        self.vBoxLayout.addSpacing(3)  
        self.vBoxLayout.addWidget(self.companyLabel)  
        
        # Description label - main text content
        self.vBoxLayout.addSpacing(20)  
        self.vBoxLayout.addWidget(self.descriptionLabel)  

        # Button layout - action buttons at bottom
        self.vBoxLayout.addSpacing(12)  
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addLayout(self.buttonLayout)
        self.buttonLayout.addWidget(self.shareButton, 0, Qt.AlignRight)  

    def openSlot(self):
        """
        Handle open button click
        
        Opens the project URL in the default web browser.
        This provides users with direct access to the project repository.
        """
        webbrowser.open(self.projectUrl)  
        # Note: Additional functionality could be added here if needed

    def shareSlot(self):
        """
        Handle share button click
        
        Copies the project URL to clipboard and shows a success notification.
        This allows users to easily share the project with others.
        """
        # Copy URL to system clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(self.projectUrl)
        
        # Show success notification with teaching tip
        TeachingTip.create(
            target=self.shareButton,  
            icon=InfoBarIcon.SUCCESS,  
            title="Success!",
            content="The project URL has been copied to your clipboard.",
            isClosable=True,  # Allow user to close manually
            tailPosition=TeachingTipTailPosition.BOTTOM,  
            duration=2000,  # Auto-hide after 2 seconds
            parent=self,
        )


class GalleryCard(HeaderCardWidget):
    """
    Gallery card widget with image flip view
    
    This card displays a horizontal flip view of images with navigation controls.
    Users can click on images to open them in a lightbox view.
    """

    def __init__(self, parent=None):
        """
        Initialize the gallery card
        
        This method sets up the gallery card with a horizontal flip view
        for displaying feature screenshots and an expand button for
        additional navigation controls.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.setTitle("Feature Showcase")  
        self.setBorderRadius(10)  

        # Create horizontal flip view for displaying images
        self.flipView = HorizontalFlipView(self)

        # Create expand button for additional controls (future feature)
        self.expandButton = TransparentToolButton(FluentIcon.CHEVRON_RIGHT_MED, self)
        self.expandButton.setFixedSize(32, 32) 
        self.expandButton.setIconSize(QSize(12, 12)) 

        # Populate flip view with sample images
        self.flipView.addImages(
            [
                ":showcase/home.png",
                ":showcase/analyze.png",
                ":showcase/simu.png",
                ":showcase/bench.png",
                ":showcase/analyze_figure.png",
                ":showcase/bench_res.png",
            ]
        )
        self.flipView.setBorderRadius(10) 
        self.flipView.setSpacing(10)
        
        # Set appropriate size for gallery display
        self.flipView.setItemSize(QSize(300, 200))  # Fixed size for gallery  

        # Add widgets to the card's layout
        self.headerLayout.addWidget(self.expandButton, 0, Qt.AlignRight)  
        self.viewLayout.addWidget(self.flipView)  


class FunctionCard(ElevatedCardWidget):
    """
    Function card widget for displaying clickable feature icons
    
    This card displays an icon and name for various application features.
    It's designed to be clickable and can trigger different actions when pressed.
    """

    def __init__(self, iconPath: str, name: str, parent=None):
        """
        Initialize the function card
        
        Args:
            iconPath (str): Path to the icon image
            name (str): Display name for the function
            parent: Parent widget
        """
        super().__init__(parent)
        iconPath = iconPath if iconPath else "GUI\\assets\\analyze.svg"  
        
        # Create icon widget for the card
        self.iconWidget = ImageLabel(iconPath)
        self.label = TitleLabel(name)  

        # Scale icon to consistent size across all cards
        self.iconWidget.scaledToHeight(128) 

        # Layout setup - vertical arrangement with icon on top, label below
        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.setAlignment(Qt.AlignCenter) 
        self.vBoxLayout.addStretch(1)  
        self.vBoxLayout.addWidget(self.iconWidget, 0, Qt.AlignCenter) 
        self.vBoxLayout.addStretch(1) 
        self.vBoxLayout.addWidget(self.label, 0, Qt.AlignHCenter | Qt.AlignBottom) 

        # Configure card appearance and interaction
        self.setFixedSize(168, 176)  
        self.setCursor(Qt.PointingHandCursor) 


class SystemRequirementCard(HeaderCardWidget):
    """
    System requirements card widget
    
    This card displays system compatibility information and requirements
    for the application, helping users understand if their system is supported.
    """

    def __init__(self, parent=None):
        """
        Initialize the system requirements card
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setTitle("System Requirements")
        self.setBorderRadius(8)

        # Information text
        self.infoLabel = BodyLabel(
            "This product is compatible with your device. Items with checkmarks meet the developer's system requirements.", self
        )
        
        # Success icon and detail link
        self.successIcon = IconWidget(InfoBarIcon.SUCCESS, self)
        self.detailButton = HyperlinkLabel("Details", self)

        # Layout setup
        self.vBoxLayout = QVBoxLayout()
        self.hBoxLayout = QHBoxLayout()

        self.successIcon.setFixedSize(16, 16)
        self.hBoxLayout.setSpacing(10)
        self.vBoxLayout.setSpacing(16)
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)

        # Add widgets to layouts
        self.hBoxLayout.addWidget(self.successIcon)
        self.hBoxLayout.addWidget(self.infoLabel)
        self.vBoxLayout.addLayout(self.hBoxLayout)
        self.vBoxLayout.addWidget(self.detailButton)

        self.viewLayout.addLayout(self.vBoxLayout)


class SettinsCard(GroupHeaderCardWidget):
    """
    Settings card widget (currently not used)
    
    This card would provide basic application settings including
    build directory, terminal display, and entry script configuration.
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

        # Control widgets
        self.chooseButton = PushButton("Choose")
        self.comboBox = ComboBox()
        self.lineEdit = SearchLineEdit()

        # Widget configuration
        self.chooseButton.setFixedWidth(120)
        self.lineEdit.setFixedWidth(320)
        self.comboBox.setFixedWidth(320)
        self.comboBox.addItems(["Always show (recommended for first packaging)", "Always hide"])
        self.lineEdit.setPlaceholderText("Enter the path to the entry script")

        # Add groups to the card
        self.addGroup(
            "resource/Rocket.svg",
            "Build Directory",
            "Select Nuitka output directory",
            self.chooseButton,
        )
        self.addGroup(
            "resource/Joystick.svg", "Terminal Display", "Set whether to show command line terminal", self.comboBox
        )
        self.addGroup(
            "resource/Python.svg", "Entry Script", "Select the application entry script", self.lineEdit
        )


class LightBox(QWidget):
    """
    Lightbox widget for displaying images in fullscreen overlay
    
    This widget provides a modal overlay for viewing images from the gallery
    with navigation controls and fade in/out animations.
    """

    def __init__(self, parent=None):
        """
        Initialize the lightbox widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent=parent)

        # Set up acrylic background effect
        if isDarkTheme():
            tintColor = QColor(32, 32, 32, 200)
        else:
            tintColor = QColor(255, 255, 255, 160)

        self.acrylicBrush = AcrylicBrush(self, 30, tintColor, QColor(0, 0, 0, 0))

        # Opacity animation setup
        self.opacityEffect = QGraphicsOpacityEffect(self)
        self.opacityAni = QPropertyAnimation(self.opacityEffect, b"opacity", self)
        self.opacityEffect.setOpacity(1)
        self.setGraphicsEffect(self.opacityEffect)

        # UI components
        self.vBoxLayout = QVBoxLayout(self)
        self.closeButton = TransparentToolButton(FluentIcon.CLOSE, self)
        self.flipView = HorizontalFlipView(self)
        self.nameLabel = BodyLabel("analyze.png", self)  # Default to first image name
        self.pageNumButton = PillPushButton("1 / 5", self)  # Updated count

        # Component configuration
        self.pageNumButton.setCheckable(False)
        self.pageNumButton.setFixedSize(80, 32)
        setFont(self.nameLabel, 16, QFont.DemiBold)

        self.closeButton.setFixedSize(32, 32)
        self.closeButton.setIconSize(QSize(14, 14))
        self.closeButton.clicked.connect(self.fadeOut)

        # Layout setup
        self.vBoxLayout.setContentsMargins(30, 32, 30, 32)
        self.vBoxLayout.addWidget(self.closeButton, 0, Qt.AlignRight | Qt.AlignTop)
        self.vBoxLayout.addWidget(self.flipView, 1)
        self.vBoxLayout.addWidget(self.nameLabel, 0, Qt.AlignHCenter)
        self.vBoxLayout.addSpacing(10)
        self.vBoxLayout.addWidget(self.pageNumButton, 0, Qt.AlignHCenter)

        # Add images to flip view with corresponding names
        self.image_names = [
            "The home window",
            "The analyze window",
            "The simulation window", 
            "The benchmark window",
            "The figure of analyze",
            "The results of benchmark",
        ]
        
        self.flipView.addImages(
            [
                ":showcase/home.png",
                ":showcase/analyze.png",
                ":showcase/simu.png",
                ":showcase/bench.png",
                ":showcase/analyze_figure.png",
                ":showcase/bench_res.png",
            ]
        )
        self.flipView.currentIndexChanged.connect(self.setCurrentIndex)

    def setCurrentIndex(self, index: int):
        """
        Set the current image index and update display
        
        This method updates both the image display and the page indicator
        to reflect the currently selected image in the lightbox.
        
        Args:
            index (int): Index of the current image (0-based)
        """
        # Update the image name/title display with actual image name
        if 0 <= index < len(self.image_names):
            image_name = self.image_names[index]
            self.nameLabel.setText(image_name)
        else:
            self.nameLabel.setText(f"Screenshot {index + 1}")
        
        # Update the page indicator button (1-based counting for user display)
        self.pageNumButton.setText(f"{index + 1} / {self.flipView.count()}")
        # Navigate to the specified image
        self.flipView.setCurrentIndex(index)

    def paintEvent(self, e):
        """
        Custom paint event for the lightbox background
        
        This method renders the background with either an acrylic effect
        (if available) or a solid color fallback for the lightbox overlay.
        
        Args:
            e: Paint event containing rendering information
        """
        # Use acrylic effect if available for modern appearance
        if self.acrylicBrush.isAvailable():
            return self.acrylicBrush.paint()

        # Fallback to solid color background
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)  

        # Choose background color based on theme
        if isDarkTheme():
            painter.setBrush(QColor(32, 32, 32))  
        else:
            painter.setBrush(QColor(255, 255, 255))  

        painter.drawRect(self.rect())  

    def resizeEvent(self, e):
        """
        Handle resize events to maintain aspect ratio
        
        This method ensures the flip view maintains the correct aspect ratio
        when the lightbox window is resized.
        
        Args:
            e: Resize event containing new size information
        """
        w = self.width() - 60  # Increased margin for better visibility
        h = self.height() - 200  # Reserve space for controls
        
        # Calculate appropriate size maintaining aspect ratio
        if w > 0 and h > 0:
            # Use the smaller dimension to ensure the image fits completely
            size = min(w, h * 16 // 9)  # 16:9 aspect ratio
            self.flipView.setItemSize(QSize(size, size * 9 // 16))

    def fadeIn(self):
        """
        Fade in animation for showing the lightbox
        
        This method creates a smooth fade-in effect when the lightbox
        is displayed, including background blur capture.
        """
        rect = QRect(self.mapToGlobal(QPoint()), self.size())
        self.acrylicBrush.grabImage(rect)

        self.opacityAni.setStartValue(0)  
        self.opacityAni.setEndValue(1)    
        self.opacityAni.setDuration(150)  
        self.opacityAni.start()
        self.show()  
    def fadeOut(self):
        """
        Fade out animation for hiding the lightbox
        
        This method creates a smooth fade-out effect when the lightbox
        is closed, with automatic hiding when animation completes.
        """
        self.opacityAni.setStartValue(1)  
        self.opacityAni.setEndValue(0)    
        self.opacityAni.setDuration(150)  
        self.opacityAni.finished.connect(self._onAniFinished)  
        self.opacityAni.start()

    def _onAniFinished(self):
        """
        Handle animation finished event
        
        This private method is called when the fade-out animation completes
        and handles the final cleanup by hiding the widget.
        """
        self.opacityAni.finished.disconnect()  
        self.hide()  


class HomeWidget(ScrollArea):
    """
    Main home widget containing all home interface components
    
    This widget serves as the main landing page for the application,
    displaying application information, feature cards, gallery, and system requirements.
    """

    def __init__(self, parent=None):
        """
        Initialize the home widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Create central view widget that will contain all interface elements
        self.view = QWidget(self)

        self.vBoxLayout = QVBoxLayout(self.view) 
        self.appCard = AppInfoCard(self) 
        self.galleryCard = GalleryCard(self)  
        
        self.funlayout = QHBoxLayout() 
        
        # Initialize function cards with icons and labels
        self.githubCard = FunctionCard(iconPath=":image/github", name="Github", parent=self)
        self.braindaCard = FunctionCard(iconPath=":image/brainda", name="Brainda", parent=self)
        self.brainstimCard = FunctionCard(iconPath=":image/brainstim", name="BrainStim", parent=self)
        self.brainflowCard = FunctionCard(iconPath=":image/brainflow", name="BrainFlow", parent=self)

        self.githubCard.setToolTip("Visit MetaBCI repository")
        self.braindaCard.setToolTip("Visit Brainda repository")
        self.brainstimCard.setToolTip("Run MetaBCI BrainStim demo")
        self.brainflowCard.setToolTip("Visit BrainFlow repository")
        
        self.githubCard.installEventFilter(ToolTipFilter(
            self.githubCard, showDelay=100, position=ToolTipPosition.TOP))
        self.braindaCard.installEventFilter(ToolTipFilter(
            self.braindaCard, showDelay=100, position=ToolTipPosition.TOP))
        self.brainstimCard.installEventFilter(ToolTipFilter(
            self.brainstimCard, showDelay=100, position=ToolTipPosition.TOP))
        self.brainflowCard.installEventFilter(ToolTipFilter(
            self.brainflowCard, showDelay=100, position=ToolTipPosition.TOP))

        self.githubCard.mousePressEvent = lambda e: webbrowser.open("https://github.com/TBC-TJU/MetaBCI")
        self.braindaCard.mousePressEvent = lambda e: webbrowser.open("https://github.com/TBC-TJU/MetaBCI/tree/master/metabci/brainda")
        self.brainflowCard.mousePressEvent = lambda e: webbrowser.open("https://github.com/TBC-TJU/MetaBCI/tree/master/metabci/brainflow")
        self.brainstimCard.mousePressEvent = lambda e: self.run_stim_demo()  

        # Add function cards to horizontal layout with top alignment
        self.funlayout.addWidget(self.githubCard, 0, Qt.AlignTop)
        self.funlayout.addWidget(self.braindaCard, 0, Qt.AlignTop)
        self.funlayout.addWidget(self.brainstimCard, 0, Qt.AlignTop)
        self.funlayout.addWidget(self.brainflowCard, 0, Qt.AlignTop)
        
        self.funcontainer = QWidget(self)
        self.funcontainer.setLayout(self.funlayout)

        self.systemCard = SystemRequirementCard(self)

        self.lightBox = LightBox(self)
        self.lightBox.hide() 
        self.galleryCard.flipView.itemClicked.connect(self.showLightBox)  # Connect click handler

        self.setWidget(self.view)  
        self.setWidgetResizable(True)  

        self.vBoxLayout.setSpacing(10)  
        self.vBoxLayout.setContentsMargins(20, 20, 20, 20) 
        self.vBoxLayout.addWidget(self.appCard, 0, Qt.AlignTop)  
        self.vBoxLayout.addWidget(self.funcontainer, 0, Qt.AlignTop)  
        self.vBoxLayout.addWidget(self.galleryCard, 0, Qt.AlignTop)  
        self.vBoxLayout.addWidget(self.systemCard, 0, Qt.AlignTop)  
        
        self.view.setAutoFillBackground(False)  
        palette = self.view.palette()
        palette.setColor(self.view.backgroundRole(), Qt.transparent)  
        self.view.setPalette(palette)
        
        # Apply custom stylesheet for seamless appearance
        self.setStyleSheet("QScrollArea{border: none; background: transparent}")

    def run_stim_demo(self):
        """
        Run the stim_demo.py script
        
        This method launches the BrainStim demonstration script in a separate process.
        It handles file existence checking and error reporting.
        """
        try:
            stim_demo_path = Path(__file__).parent / "stim_demo.py"

            if not stim_demo_path.exists():
                from PySide6.QtWidgets import QMessageBox
                InfoBar.error(
                    title="Error",
                    content=f"File not found:\n{stim_demo_path}",
                    orient=Qt.Vertical,  
                    isClosable=True,  
                    position=InfoBarPosition.BOTTOM_RIGHT,  
                    duration=-1, 
                    parent=self,
                )
                return

            subprocess.Popen([sys.executable, str(stim_demo_path)])

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Launch Failed", f"Error running stim_demo.py:\n{str(e)}")

    def showLightBox(self):
        """
        Show the lightbox with the current gallery image
        
        This method displays the lightbox overlay for viewing gallery images
        in fullscreen mode with navigation controls.
        """
        index = self.galleryCard.flipView.currentIndex()
        self.lightBox.setCurrentIndex(index)
        self.lightBox.fadeIn()

    def resizeEvent(self, e):
        """
        Handle resize events to update lightbox size
        
        This ensures the lightbox always covers the entire widget area
        when the window is resized.
        
        Args:
            e: Resize event containing new size information
        """
        super().resizeEvent(e)  
        self.lightBox.resize(self.size())  
