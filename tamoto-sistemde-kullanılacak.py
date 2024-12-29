import cv2
import numpy as np
import serial
from collections import deque
from multiprocessing import shared_memory
import time


class BalloonDetector:
    def __init__(self, arduino_port=None, frame_shape=(480, 640, 3)):
        self.red_points = deque(maxlen=32)
        self.blue_points = deque(maxlen=32)
        self.hsv_values = {
            'red_low_h': 0, 'red_low_s': 120, 'red_low_v': 70,
            'red_high_h': 10, 'red_high_s': 255, 'red_high_v': 255,
            'blue_low_h': 100, 'blue_low_s': 120, 'blue_low_v': 70,
            'blue_high_h': 130, 'blue_high_s': 255, 'blue_high_v': 255
        }
        self.arduino_port = arduino_port
        self.arduino = None
        self.frame_shape = frame_shape

        # Ekranın ortasındaki nokta
        self.center_point = (frame_shape[1] // 2, frame_shape[0] // 2)  # (x, y)

        # Zamanlayıcı ve atış durumu
        self.shot_timer = None
        self.shot_triggered = False

        # Ortak bellekleri temizle ve oluştur
        self.cleanup_shared_memory("raw_frame")
        self.cleanup_shared_memory("processed_frame")
        self.raw_shm = self.get_or_create_shared_memory("raw_frame", frame_shape)
        self.processed_shm = self.get_or_create_shared_memory("processed_frame", frame_shape)

        self.raw_buffer = np.ndarray(frame_shape, dtype='uint8', buffer=self.raw_shm.buf)
        self.processed_buffer = np.ndarray(frame_shape, dtype='uint8', buffer=self.processed_shm.buf)

        # Trackbar'ları oluştur
        self.create_trackbars()

        if self.arduino_port:
            self.open_arduino()
            time.sleep(2)  # Arduino'nun hazır olması için bekleyin
            self.send_data("3\r")  # Başlangıçta Arduino'ya 3 gönder

    def cleanup_shared_memory(self, name):
        """Önceden var olan paylaşımlı belleği temizle."""
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
            print(f"Paylaşımlı bellek '{name}' temizlendi.")
        except FileNotFoundError:
            print(f"Paylaşımlı bellek '{name}' bulunamadı, temizleme gerekmiyor.")
        except Exception as e:
            print(f"Paylaşımlı bellek temizleme hatası: {e}")

    def get_or_create_shared_memory(self, name, shape):
        """Paylaşımlı bellek oluştur veya var olanı bağla."""
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=int(np.prod(shape) * np.dtype('uint8').itemsize))
            print(f"Yeni paylaşımlı bellek '{name}' oluşturuldu.")
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=name)
            print(f"Mevcut paylaşımlı bellek '{name}' ile bağlandı.")
        return shm

    def create_trackbars(self):
        """HSV değerlerini ayarlamak için trackbar'lar oluştur."""
        cv2.namedWindow("Kontroller")
        for key in self.hsv_values:
            cv2.createTrackbar(key, "Kontroller", self.hsv_values[key], 255, lambda x: None)

    def get_trackbar_values(self):
        """Trackbar'lardan HSV değerlerini al."""
        values = {}
        for key in self.hsv_values:
            values[key] = cv2.getTrackbarPos(key, "Kontroller")
        return values

    def open_arduino(self):
        """Arduino bağlantısını aç."""
        if not self.arduino and self.arduino_port:
            try:
                self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
                print(f"Arduino bağlantısı {self.arduino_port} üzerinde açıldı.")
            except serial.SerialException as e:
                print(f"Arduino bağlantısı açılamadı: {e}")

    def send_data(self, data):
        """Arduino'ya veri gönder."""
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(data.encode('utf-8'))
                print(f"Arduino'ya veri gönderildi: {data.strip()}")
            except Exception as e:
                print(f"Arduino'ya veri gönderme hatası: {e}")

    def close_arduino(self):
        """Arduino bağlantısını kapat."""
        if self.arduino:
            self.arduino.close()
            self.arduino = None
            print("Arduino bağlantısı kapatıldı.")

    def preprocess_frame(self, frame):
        """Görüntü ön işleme."""
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def detect_balloons(self, frame):
        """Kırmızı ve mavi balonları tespit et."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        values = self.get_trackbar_values()

        lower_red1 = np.array([values['red_low_h'], values['red_low_s'], values['red_low_v']])
        upper_red1 = np.array([values['red_high_h'], values['red_high_s'], values['red_high_v']])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, values['red_low_s'], values['red_low_v']])
        upper_red2 = np.array([180, values['red_high_s'], values['red_high_v']])
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        lower_blue = np.array([values['blue_low_h'], values['blue_low_s'], values['blue_low_v']])
        upper_blue = np.array([values['blue_high_h'], values['blue_high_s'], values['blue_high_v']])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        return mask_red, mask_blue

    def draw_bounding_boxes(self, frame, mask_red, mask_blue):
        """Kırmızı ve mavi balonlar için kutucuklar çiz."""
        for contour in cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                self.red_points.appendleft(center)

                # Koordinatları sadece belirli bir mesafe değiştiğinde gönder
                if len(self.red_points) > 1:
                    prev_center = self.red_points[1]
                    distance = np.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
                    if distance > 5:  # Mesafe 20 pikselden fazla değiştiyse gönder
                        self.send_data(f"{center[0]},{center[1]}\r\n")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Red Balloon", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Bounding box ekranın ortasındaki noktayı kapsıyorsa
                if self.is_center_inside_box(x, y, w, h, tolerance=50):  # Tolerans değeri artırıldı
                    if self.shot_timer is None:
                        self.shot_timer = time.time()  # Zamanlayıcıyı başlat
                    elif time.time() - self.shot_timer >= 0.3 and not self.shot_triggered:  # Süre kısaltıldı
                        self.send_data("4\r\n")  # Ateş komutunu gönder
                        print("Ateşşşş!")  # Terminale "Ateşşşş!" yazdır
                        self.shot_triggered = True
                else:
                    self.shot_timer = None
                    self.shot_triggered = False

        for contour in cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                self.blue_points.appendleft(center)

                # Koordinatları sadece belirli bir mesafe değiştiğinde gönder
                if len(self.blue_points) > 1:
                    prev_center = self.blue_points[1]
                    distance = np.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
                    if distance > 10:  # Mesafe 20 pikselden fazla değiştiyse gönder
                        self.send_data(f"{center[0]},{center[1]}\r\n")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Blue Balloon", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame

    def is_center_inside_box(self, x, y, w, h, tolerance=50):
        """Ekranın ortasındaki nokta bounding box içinde mi? (Tolerans değeri artırıldı)"""
        return (x - tolerance <= self.center_point[0] <= x + w + tolerance and
                y - tolerance <= self.center_point[1] <= y + h + tolerance)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Kamera açılamadı!")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Görüntü alınamadı!")
                    break

                # İşlenmemiş görüntüyü ortak belleğe yaz
                np.copyto(self.raw_buffer, frame)

                processed_frame = self.preprocess_frame(frame)
                mask_red, mask_blue = self.detect_balloons(processed_frame)
                frame_with_boxes = self.draw_bounding_boxes(processed_frame, mask_red, mask_blue)

                # Ekranın ortasına nokta çiz
                cv2.circle(frame_with_boxes, self.center_point, 5, (0, 255, 0), -1)

                # İşlenmiş görüntüyü ortak belleğe yaz
                np.copyto(self.processed_buffer, frame_with_boxes)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            self.close_arduino()
            self.raw_shm.close()
            self.raw_shm.unlink()
            self.processed_shm.close()
            self.processed_shm.unlink()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = BalloonDetector(arduino_port="COM6")
    detector.run()