import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import playsound
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def getCoordinates_fp(face_landmarks, img_h, img_w):
    face_3d = []
    face_2d = []
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)
    # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64)
    return face_2d, face_3d, nose_2d, nose_3d

def projectCameraAngle_fp(face_2d, face_3d, img_h, img_w):
    # The camera matrix
    focal_length = 1 * img_w
    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])
    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    return x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix

def getHeadTilt_fp(x, y, z):
    if y < -5:
        tiltPose = "Left"
    elif y > 5:
        tiltPose = "Right"
    elif x < -15:
        tiltPose = "Down"
    elif x > 20:
        tiltPose = "Up"
    else:
        tiltPose = "Forward"
    return tiltPose


WARNING_STATE = 0


def draw_head_tilt_pose_fp(image, text, nose_2d):
    global WARNING_STATE
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    if text == "Right":
        WARNING_STATE = 1
        result = "WARNING DON'T LOOK RIGHT"
    elif text == "Left":
        WARNING_STATE = 1
        result = "WARNING DON'T LOOK LEFT"
    else:
        WARNING_STATE = 0
        result = "DETECTED"

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2

    text_width, text_height = cv2.getTextSize(result, fontFace, fontScale, thickness)[0]

    CenterCoordinates = (int(p1[0]) - int(text_width / 2), int(p1[1] - 100) + int(text_height / 2))

    cv2.putText(image, f"{result}", CenterCoordinates, fontFace, fontScale,
                (0, 255, 0), 2)
    return WARNING_STATE


def pipelineHeadTiltPose(image, face_landmarks):
    # Get image shape
    img_h, img_w, img_c = image.shape
    # Get face features coordinate
    face_2d, face_3d, nose_2d, nose_3d = getCoordinates_fp(face_landmarks, img_h, img_w)
    # Get camera angle
    x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix = projectCameraAngle_fp(face_2d, face_3d, img_h, img_w)
    # Get head tilt
    head_pose = getHeadTilt_fp(x, y, z)

    warning = draw_head_tilt_pose_fp(image, head_pose, nose_2d)
    return warning
