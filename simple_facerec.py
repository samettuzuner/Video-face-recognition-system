import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = [] # tanınan yüzlerin öğrenilmiş kodlarının listesi
        self.known_face_names = [] # tanınan yüzlerin isimlerinin listesi

        # Daha hızlı işlem yapabilmek için çerçeveleri yeniden boyutlandırır
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Yolu verilen klasördeki kodlama görüntülerini yükler.
        :param images_path:
        :return:
        """
        # Görüntüleri yükle
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} kodlama görüntüsü bulundu.".format(len(images_path)))

        # İmaj kodlamalarını ve isimlerini sakla
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # İlk dosya yolundan yalnızca dosya adını alın.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Kodlamayı al
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Dosya adını ve dosya kodlamasını sakla
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Kodlama görüntüleri yüklendi")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Video görüntüsündeki tüm yüzleri ve yüz kodlamalarını bulun
        # OpenCV'nin kullandığı BGR renkten (OpenCV kullanır) face_recognition'ın kullandığı RGB renge dönüştürün
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Yüzün öğrenilmiş yüzlerden biriyle eşleşip eşleşmediğini kontrol edin
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # Eğer bir eşleşme known_face_encodings içinde bulunduysa, sadece ilkini kullan.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Veya bunun yerine, yeni yüze en küçük uzaklığı olan tanınan yüzü kullanın
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Çerçeve yeniden boyutlandırması ile numpy dizisine dönüştürür.
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
