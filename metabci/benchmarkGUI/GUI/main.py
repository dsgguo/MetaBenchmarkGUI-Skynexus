# coding:utf-8
from PySide6.QtCore import QSize, QEventLoop, QTimer, Qt
from PySide6.QtGui import QIcon, QPixmap, QFont
from PySide6.QtWidgets import QApplication, QListWidget, QWidget, QHBoxLayout, QLabel

from qfluentwidgets import SplashScreen, FluentWindow, FluentTranslator
from qfluentwidgets import FluentIcon 

import resource_rc

from analyze_window import AnalyzeWidget
from home_window import HomeWidget
from simu_window import SimuWidget
from benchmark_window import BenchmarkWidget

import sys
class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.resize(1000, 700)
        self.setWindowTitle("MetaBenchmark-GUI")
        self.setWindowIcon(
            QIcon(":/icon/Skynexus2.png"
            )
        )

        Splashicon = QIcon(":/icon/Skynexus.png")
        self.splashScreen = SplashScreen(Splashicon, self)
        self.splashScreen.setIconSize(QSize(512, 512))

        self.show()

        self.analyzeWidget = AnalyzeWidget(self)
        self.homeWidget = HomeWidget(self)
        self.simuWidget = SimuWidget(self)
        self.benchmarkWidget = BenchmarkWidget(self)
  
        self.analyzeWidget.setObjectName("analyzeWidget")
        self.homeWidget.setObjectName("homeWidget")
        self.simuWidget.setObjectName("simuWidget")
        self.benchmarkWidget.setObjectName("benchmarkWidget")

        self.addSubInterface(self.homeWidget, icon=FluentIcon.HOME, text="Home"
                             )
        self.addSubInterface(self.analyzeWidget, icon=QIcon(":/icon/analyze.svg"), text="Analyze"
                             )
        self.addSubInterface(self.simuWidget, icon=QIcon(":/icon/SimuOnline.svg"), text="Online Simulation"
                             )        
        self.addSubInterface(self.benchmarkWidget, icon=QIcon(":/icon/benchmark.svg"), text="Benchmark"
                             )

        self.createSplashScreen()

    def createSplashScreen(self):
        loop = QEventLoop(self)
        QTimer.singleShot(1500, loop.quit)
        loop.exec()
        self.splashScreen.finish()


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    translator = FluentTranslator()
    app.installTranslator(translator)
    
    w = MainWindow()
    w.show()
    app.exec()
