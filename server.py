from PySide6.QtCore import QRect, QCoreApplication, Qt, QThread, Signal, QObject, QRunnable, Slot, QThreadPool, QPoint, \
    QSize, QMargins
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import *

from datetime import datetime

import cv2
import imagezmq

class FlowLayout(QLayout):
    def __init__(self, parent=None):
        super().__init__(parent)

        if parent is not None:
            self.setContentsMargins(QMargins(0, 0, 0, 0))

        self._item_list = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]

        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)

        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._do_layout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()

        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())

        size += QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._item_list:
            style = item.widget().style()
            layout_spacing_x = style.layoutSpacing(
                QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Orientation.Horizontal
            )
            layout_spacing_y = style.layoutSpacing(
                QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Vertical
            )
            space_x = spacing + layout_spacing_x
            space_y = spacing + layout_spacing_y
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()

print("Starting Server...")

image_hub = imagezmq.ImageHub()

def decomp_img(img):
    decimg = cv2.imdecode(img, 1)
    return decimg


lastActive = {}
clientImages = {}
clientImagesWidget = {}
connectedClients = []

print("Server Started....")

class ServerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("ExamGuard Server")
        MainWindow.resize(1201, 751)

        self.centralwidget = QWidget(MainWindow)

        self.layout = QGridLayout()

        self.threadpool = QThreadPool()
        server_worker = ServerWorker()
        server_worker.signals.updateImageFeed.connect(self.setVideoFeed)
        self.threadpool.start(server_worker)

        self.centralwidget.setLayout(self.layout)
        MainWindow.setCentralWidget(self.centralwidget)

    def setVideoFeed(self, clientName):
        if clientName not in connectedClients:
            VideoFeed = QLabel("")
            VideoFeed.setGeometry(QRect(670, 30, 481, 371))
            VideoFeed.setPixmap(QPixmap.fromImage(clientImages[clientName]))
            self.layout.addWidget(VideoFeed)
            clientImagesWidget[clientName] = VideoFeed
            connectedClients.append(clientName)
        clientImagesWidget[clientName].setPixmap(QPixmap.fromImage(clientImages[clientName]))

class ServerWorkerSignals(QObject):
    updateImageFeed = Signal(str)

class ServerWorker(QRunnable):
    def __init__(self):
        super(ServerWorker, self).__init__()

        self.signals = ServerWorkerSignals()

    @Slot()
    def run(self):
        while True:
            rpi_name, image = image_hub.recv_image()
            image_hub.send_reply(b'OK')
            if rpi_name not in lastActive.keys():
                print("[INFO] receiving data from {}...".format(rpi_name))

            decomp = decomp_img(image)
            convert_to_qt = QImage(decomp.data, decomp.shape[1], decomp.shape[0], QImage.Format.Format_BGR888)
            pic = convert_to_qt.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)

            lastActive[rpi_name] = datetime.now()
            clientImages[rpi_name] = pic
            self.signals.updateImageFeed.emit(rpi_name)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    AppMainWindow = QMainWindow()
    ui = ServerWindow()
    ui.setupUi(AppMainWindow)
    AppMainWindow.show()
    sys.exit(app.exec_())
