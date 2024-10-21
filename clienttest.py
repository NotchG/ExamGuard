from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import mediapipe as mp
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from originalscripts.g_helper import mirrorImage
from FaceMeshDetector import pipelineHeadTiltPose
import socket
import imagezmq

# Initialize Face Mesh #########################
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
################################################

# Initialize Camera #########################
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
##############################################

# SERVER CONNECT #########################
sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')
hostname = socket.gethostname()
##########################################

class Ui_MainWindow(object):

    def __init__(self):
        self.WARNING_STATE = 0
        self.warning_count = 0

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1201, 751)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.VideoFeed = QtWidgets.QLabel(self.centralwidget)
        self.VideoFeed.setGeometry(QtCore.QRect(670, 30, 481, 371))
        self.VideoFeed.setText("")
        self.VideoFeed.setObjectName("VideoFeed")

        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.AIUpdate.connect(self.AIUpdateSlot)

        self.ExamGuardLabel = QtWidgets.QLabel(self.centralwidget)
        self.ExamGuardLabel.setGeometry(QtCore.QRect(20, 30, 619, 71))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.ExamGuardLabel.setFont(font)
        self.ExamGuardLabel.setObjectName("ExamGuardLabel")

        self.WarningLabel = QtWidgets.QLabel(self.centralwidget)
        self.WarningLabel.setGeometry(QtCore.QRect(20, 110, 619, 91))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.WarningLabel.setFont(font)
        self.WarningLabel.setObjectName("WarningLabel")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "EXAM GUARD"))
        self.ExamGuardLabel.setText(_translate("MainWindow", "EXAM GUARD"))
        self.WarningLabel.setText(_translate("MainWindow", "Warnings: 0"))

    def AIUpdateSlot(self, image, warningState):
        self.VideoFeed.setPixmap(QPixmap.fromImage(image))
        if warningState == 1 and self.WARNING_STATE != warningState:
            self.warning_count += 1
            self.WarningLabel.setText(f"Warnings: {self.warning_count}")
            self.WARNING_STATE = 1
        elif warningState == 0 and self.WARNING_STATE != warningState:
            self.WARNING_STATE = 0

    def closeEvent(self, event):
        print("Close clicked")
        sender.close()
        event.accept()

class Worker1(QThread):
    AIUpdate = pyqtSignal(QImage, int)

    def __init__(self):
        super().__init__()
        self.thread_active = True

    def run(self):
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

                    sender.send_image(hostname, image)  # Send Image to server
                    convert_to_qt = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_BGR888)
                    pic = convert_to_qt.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                    self.AIUpdate.emit(pic, head_tilt_pose)
        cap.release()

    def stop(self):
        self.thread_active = False
        self.quit()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
