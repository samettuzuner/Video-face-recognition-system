import cv2
from simple_facerec import SimpleFacerec

# Yüz tanıma sınıfını yükleme ve yüzleri kodlama
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Video dosyasının yüklenmesi
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    # Eğer video bitti ise döngüyü sonlandırma
    if not ret:
        break

    # Tanınmış yüzleri tespit etme
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Yüz isimlerini ve dikdörtgen sınırları çizme
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # İşlenen karenin gösterimi
    cv2.imshow("Frame", frame)

    # 'Esc' tuşuna basarak programı sonlandırma
    key = cv2.waitKey(1)
    if key == 27:
        break

# Video akışını serbest bırakma ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
