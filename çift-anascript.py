import serial
import time
from multiprocessing import shared_memory
import subprocess

# Seri port bilgisi
SERIAL_PORT = "COM6"
BAUD_RATE = 9600

# Ortak hafıza adı ve boyutu
SHARED_MEMORY_NAME = "SystemMode"
SHARED_MEMORY_SIZE = 10  # En fazla 10 karakterlik veri

# Modlara karşılık gelen script yolları
SCRIPTS = {
    "1": "C:/Users/skyks/Desktop/yolov7-main/otonom.py",
    "2": "C:/Users/skyks/Desktop/yolov7-main/otonomv2memory.py",
    "3": "C:/Users/skyks/Desktop/yolov7-main/otonom.py"
}

# Global değişkenler
arduino = None
current_mode = None
current_process = None

def cleanup_shared_memory(name):
    """Belirtilen paylaşımlı belleği temizler."""
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
        print(f"Paylaşımlı bellek '{name}' temizlendi.")
    except FileNotFoundError:
        print(f"Paylaşımlı bellek '{name}' bulunamadı.")
    except Exception as e:
        print(f"Paylaşımlı bellek temizleme hatası: {e}")

def open_arduino():
    """Arduino bağlantısını açar."""
    global arduino
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Seri port {SERIAL_PORT} başarıyla açıldı.")
    except serial.SerialException as e:
        print(f"Arduino bağlantısı açılamadı: {e}")
        arduino = None

def close_arduino():
    """Arduino bağlantısını kapatır."""
    global arduino
    if arduino:
        try:
            arduino.close()
            print("Seri port kapatıldı.")
        except Exception as e:
            print(f"Seri port kapatma hatası: {e}")
        finally:
            arduino = None

def read_from_arduino():
    """Arduino'dan veri okur."""
    global arduino
    if arduino is None:
        return None
    if arduino.in_waiting > 0:
        try:
            data = arduino.readline().decode('utf-8', errors='ignore').strip()
            return data
        except Exception as e:
            print(f"Arduino'dan veri okuma hatası: {e}")
    return None

def write_to_shared_memory(value):
    """Shared memory'ye veri yaz."""
    try:
        shm = shared_memory.SharedMemory(name=SHARED_MEMORY_NAME)
        data = value.ljust(SHARED_MEMORY_SIZE)  # Veriyi shared memory boyutuna uygun hale getir
        shm.buf[:len(data)] = bytes(data, "utf-8")
    except Exception as e:
        print(f"Shared memory'ye yazma hatası: {e}")

def read_from_shared_memory():
    """Shared memory'den veri oku."""
    try:
        shm = shared_memory.SharedMemory(name=SHARED_MEMORY_NAME)
        data = bytes(shm.buf[:SHARED_MEMORY_SIZE]).decode("utf-8").strip()
        return data
    except Exception as e:
        print(f"Shared memory'den okuma hatası: {e}")
        return None

def stop_current_process():
    """Çalışan scripti durdur."""
    global current_process
    if current_process and current_process.poll() is None:  # Process çalışıyorsa
        current_process.terminate()
        print("Mevcut script durduruldu.")
        current_process = None
        # Paylaşımlı bellekleri temizle
        cleanup_shared_memory("raw_frame")
        cleanup_shared_memory("processed_frame")

def start_new_process(mode):
    """Yeni scripti başlat."""
    global current_process
    script_path = SCRIPTS.get(mode)
    if script_path:
        print(f"Yeni script çalıştırılıyor: {script_path}")
        current_process = subprocess.Popen(["python", script_path])
    else:
        print(f"Bilinmeyen mod: {mode}")

def send_data_to_arduino(data):
    """Arduino'ya veri gönderme fonksiyonu."""
    global arduino
    if arduino:
        try:
            # "/r" (carriage return) ekle
            data_with_carriage_return = data + "\r"
            arduino.write(data_with_carriage_return.encode('utf-8'))
            print(f"Arduino'ya veri gönderildi: {data_with_carriage_return}")
        except Exception as e:
            print(f"Arduino'ya veri gönderme hatası: {e}")
    else:
        print("Arduino bağlantısı yok, veri gönderilemedi.")

def main():
    global current_mode, current_process

    # Ortak hafızayı aç veya oluştur
    try:
        shm = shared_memory.SharedMemory(name=SHARED_MEMORY_NAME, create=True, size=SHARED_MEMORY_SIZE)
        write_to_shared_memory("Bekleniyor")  # Varsayılan değer
        print("Ortak hafıza oluşturuldu ve varsayılan değer yazıldı.")
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=SHARED_MEMORY_NAME, create=False)
        print("Ortak hafıza zaten mevcut.")

    # Arduino bağlantısını aç
    open_arduino()
    print("Sistem başlatıldı, veri bekleniyor...")

    try:
        while True:
            # Arduino'dan veri al
            arduino_data = read_from_arduino()
            if arduino_data and arduino_data in SCRIPTS:
                print(f"Arduino'dan gelen veri: {arduino_data}")
                current_mode = arduino_data
                # Arduino'ya modu gönder
                send_data_to_arduino(current_mode)
                # İlgili scripti çalıştır
                start_new_process(current_mode)

            # Shared memory'den veri al
            shared_memory_data = read_from_shared_memory()
            if shared_memory_data and shared_memory_data in SCRIPTS:
                print(f"Shared memory'den gelen veri: {shared_memory_data}")
                if shared_memory_data != current_mode:
                    print(f"Mod değişikliği tespit edildi: {current_mode} -> {shared_memory_data}")
                current_mode = shared_memory_data
                # Arduino'ya modu gönder
                send_data_to_arduino(current_mode)
                # İlgili scripti çalıştır
                start_new_process(current_mode)

            time.sleep(1)  # Döngü gecikmesi
    except KeyboardInterrupt:
        print("Program sonlandırılıyor...")
    finally:
        stop_current_process()
        shm.close()
        shm.unlink()  # Ortak hafızayı temizle
        close_arduino()

if __name__ == "__main__":
    main()