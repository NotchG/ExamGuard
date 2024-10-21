from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import mediapipe as mp
from originalscripts.g_helper import mirrorImage
from originalscripts.fp_helper import pipelineHeadTiltPose
import sys
import socket
import imagezmq


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


# Initiate Camera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')
hostname = socket.gethostname()

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("ExamGuard Client")
        self.setGeometry(0, 0, 1280, 720)
        self.vbox = QVBoxLayout()

        self.feed_label = QLabel("ExamGuard Video")
        self.vbox.addWidget(self.feed_label)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.CancelFeed)  # type: ignore
        self.vbox.addWidget(self.cancel_btn)

        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        self.vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setLayout(self.vbox)

    def ImageUpdateSlot(self, Image):
        self.feed_label.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.thread_active = True
        while self.thread_active:
            with mp_face_mesh.FaceMesh(max_num_faces=30, refine_landmarks=True, min_detection_confidence=0.5,
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
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            # HEAD TILT POSE -----------------------------------
                            head_tilt_pose = pipelineHeadTiltPose(image, face_landmarks)

                    sender.send_image(hostname, image)
                    convert_to_qt = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_BGR888)
                    pic = convert_to_qt.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                    self.ImageUpdate.emit(pic)
        cap.release()

    def stop(self):
        self.thread_active = False
        self.quit()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()