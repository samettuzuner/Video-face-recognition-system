import cv2  # OpenCV kütüphanesi import ediliyor.
from simple_facerec import SimpleFacerec  # simple_facerec kütüphanesinden SimpleFacerec sınıfı import ediliyor.
from datetime import datetime, date  # datetime ve date sınıfları import ediliyor.

# Bugünün tarihini alma işlemi yapılıyor.
today = date.today()
day = today.strftime("%b-%d-%Y")
day_str = "Yoklama_Raporu-" + day + ".csv"
print(day_str)

# Tarih isimli dosya oluşturuluyor ve dosya adı print ediliyor.
dosya = open(day_str, "a")
dosya.write("Ad, Saat")
dosya.close()

# Öğrenci isimlerini ve yoklama bilgilerini yazan fonksiyon tanımlanıyor.
def yoklamayaYaz(name):
    with open(day_str, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Resim dosyalarındaki yüzleri tanıyıp kodlayan sınıf objesi oluşturuluyor.
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Videoyu projeye dahil etme
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()  # Video kareleri tek tek okunuyor.

    # Yüz tespiti yapılıyor.
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Yüz tespit edilen kişinin ismi kare üzerine yazdırılıyor.
        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

        # Yüz çerçevesi çizdiriliyor.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Öğrenci isimlerini ve yoklama bilgilerini yazan fonksiyon çağırılıyor.
        yoklamayaYaz(name)

    # Frame penceresi gösteriliyor.
    cv2.imshow("Frame", frame)

    # ESC tuşuna basılırsa döngüden çıkılıyor.
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()  # Kamera kapatılıyor.
cv2.destroyAllWindows()  # Pencereler kapatılıyor.
