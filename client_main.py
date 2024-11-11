from PySide6.QtCore import QRect, QCoreApplication, Qt, QThread, Signal, QObject, QRunnable, Slot, QThreadPool
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import *

import cv2
import mediapipe as mp
import zmq
import ctypes

from FaceMeshDetector import pipelineHeadTiltPose, mirrorImage
import socket
import imagezmq

# Initialize Face Mesh #########################
print("Initializing Face Mesh...")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
################################################

# Initialize Camera #########################
print("Initializing Camera....")
cap = cv2.VideoCapture(0)

width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
##############################################

# SERVER CONNECT #########################
ip_addr = input("Please input server ip: ")
print("Connecting to server....")
sender = imagezmq.ImageSender(connect_to=f'tcp://{ip_addr}:5555')
# context = zmq.Context()
# textSender = context.socket(zmq.REQ)
# textSender.connect("tcp://localhost:5556")
hostname = socket.gethostname()
##########################################


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.WARNING_STATE = 0
        self.warning_count = 0

    def setupUi(self, MainWindow):
        print("Setting Up UI")

        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("ExamGuard Client")
        MainWindow.resize(1201, 751)

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.VideoFeed = QLabel(self.centralwidget)
        self.VideoFeed.setGeometry(QRect(670, 30, 481, 371))
        self.VideoFeed.setText("")
        self.VideoFeed.setObjectName("VideoFeed")

        print("Setting Up ThreadPool")

        self.threadpool = QThreadPool()
        ai_worker = AIWorker()
        ai_worker.signals.imageFeed.connect(self.setVideoFeed)
        ai_worker.signals.warningState.connect(self.setWarningState)
        self.threadpool.start(ai_worker)

        self.ExamGuardLabel = QLabel(self.centralwidget)
        self.ExamGuardLabel.setGeometry(QRect(20, 30, 619, 71))
        font = QFont()
        font.setPointSize(30)
        self.ExamGuardLabel.setFont(font)
        self.ExamGuardLabel.setObjectName("ExamGuardLabel")

        self.WarningLabel = QLabel(self.centralwidget)
        self.WarningLabel.setGeometry(QRect(20, 110, 619, 91))
        font = QFont()
        font.setPointSize(30)
        self.WarningLabel.setFont(font)
        self.WarningLabel.setObjectName("WarningLabel")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        print("Finished Setting Up UI")

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "EXAM GUARD"))
        self.ExamGuardLabel.setText(_translate("MainWindow", "EXAM GUARD"))
        self.WarningLabel.setText(_translate("MainWindow", "Warnings: 0"))

    def setWarningState(self, warningState):
        if warningState == 1 and self.WARNING_STATE != warningState:
            self.warning_count += 1
            self.WarningLabel.setText(f"Warnings: {self.warning_count}")
            self.WARNING_STATE = 1
        elif warningState == 0 and self.WARNING_STATE != warningState:
            self.WARNING_STATE = 0

    def setVideoFeed(self, image):
        self.VideoFeed.setPixmap(QPixmap.fromImage(image))



def compress_image(img, quality=90):
    # Encode the image with JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)

    return encimg


class AIWorkerSignals(QObject):
    imageFeed = Signal(QImage)
    warningState = Signal(int)


class AIWorker(QRunnable):
    def __init__(self):
        super(AIWorker, self).__init__()

        print("AI Initializing")

        self.signals = AIWorkerSignals()
        self.thread_active = True

    @Slot()
    def run(self):
        print("AI Started")
        while self.thread_active:
            with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5) as face_mesh:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        continue

                    # Mirror image (Optional)
                    image = mirrorImage(image)

                    # Generate face mesh
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    # Processing Face Landmarks
                    head_tilt_pose = 0
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            # HEAD TILT POSE -----------------------------------
                            head_tilt_pose = pipelineHeadTiltPose(image, face_landmarks)

                    compressed_img = compress_image(image, 40)

                    sender.send_image(hostname, compressed_img)  # Send Image to server
                    convert_to_qt = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_BGR888)
                    pic = convert_to_qt.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                    self.signals.imageFeed.emit(pic)
                    self.signals.warningState.emit(head_tilt_pose)
        cap.release()

    def stop(self):
        self.thread_active = False
        self.quit()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    AppMainWindow = QMainWindow()
    ui = MainWindow()
    ui.setupUi(AppMainWindow)
    AppMainWindow.show()
    sys.exit(app.exec_())
