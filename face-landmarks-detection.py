import cv2
import numpy as np
import dlib

# Video akışı yakalama
cap = cv2.VideoCapture(0)

# Yüz tespitçisi oluşturma
detector = dlib.get_frontal_face_detector()

# Yüz işaretleyici oluşturma
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Video akışından karelerin işlenmesi
while True:
    _, frame = cap.read()

    # Gri tonlamalı görüntü oluşturma
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Yüzün dikdörtgen sınırlarının çizimi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Yüz işaretleyicilerinin tespiti ve çizimi
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

    # İşlenen karenin gösterimi
    cv2.imshow("Frame", frame)

    # 'Esc' tuşuna basarak programı sonlandırma
    key = cv2.waitKey(1)
    if key == 27:
        break
