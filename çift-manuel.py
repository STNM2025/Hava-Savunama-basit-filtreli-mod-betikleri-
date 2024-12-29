import cv2
import numpy as np
from multiprocessing import shared_memory

class BalloonDetector:
    def __init__(self, frame_shape=(480, 640, 3)):
        self.frame_shape = frame_shape

        # Ekranın ortasındaki nokta
        self.center_point = (frame_shape[1] // 2, frame_shape[0] // 2)  # (x, y)

        # Ortak bellekleri temizle ve oluştur
        self.cleanup_shared_memory("raw_frame")
        self.cleanup_shared_memory("processed_frame")
        self.raw_shm = self.get_or_create_shared_memory("raw_frame", frame_shape)
        self.processed_shm = self.get_or_create_shared_memory("processed_frame", frame_shape)

        self.raw_buffer = np.ndarray(frame_shape, dtype='uint8', buffer=self.raw_shm.buf)
        self.processed_buffer = np.ndarray(frame_shape, dtype='uint8', buffer=self.processed_shm.buf)

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

    def process_frame(self, frame):
        """Görüntünün ortasına bir nokta çiz."""
        processed_frame = frame.copy()
        cv2.circle(processed_frame, self.center_point, 5, (0, 255, 0), -1)  # Ortaya yeşil bir nokta çiz
        return processed_frame

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

                # Ham görüntüyü ortak belleğe yaz
                np.copyto(self.raw_buffer, frame)

                # Görüntüyü işle (ortasına nokta çiz)
                processed_frame = self.process_frame(frame)

                # İşlenmiş görüntüyü ortak belleğe yaz
                np.copyto(self.processed_buffer, processed_frame)

                # Görüntüleri göster (isteğe bağlı)
                cv2.imshow("Raw Frame", frame)
                cv2.imshow("Processed Frame", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            self.raw_shm.close()
            self.raw_shm.unlink()
            self.processed_shm.close()
            self.processed_shm.unlink()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = BalloonDetector()
    detector.run()