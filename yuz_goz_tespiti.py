import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class FaceEyeDetector:
    def __init__(self):
        """
        Yüz ve Göz Tespit Sistemi Başlatıcı
        
        Haar Cascade sınıflandırıcılarını yükler:
        - Yüz tespiti için
        - Göz tespiti için
        - Gülümseme tespiti için
        - Profil yüz tespiti için
        """
        
        # Kamerayı başlat (0: varsayılan kamera)
        self.cap = cv2.VideoCapture(0)
        
        # Cascade dosyalarını yükle
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        
        # Cascade'lerin yüklendiğini kontrol et
        cascades = {
            "Yüz": self.face_cascade,
            "Göz": self.eye_cascade,
            "Gülümseme": self.smile_cascade,
            "Profil": self.profile_cascade
        }
        
        print("="*50)
        print("CASCADE YÜKLEME DURUMU")
        print("="*50)
        
        for name, cascade in cascades.items():
            if cascade.empty():
                print(f"✗ {name} cascade yüklenemedi!")
            else:
                print(f"✓ {name} cascade yüklendi")
        
        # Tespit modları
        self.detection_modes = {
            "face_only": "Sadece Yüz",
            "face_eyes": "Yüz + Göz", 
            "face_smile": "Yüz + Gülümseme",
            "profile": "Profil Yüz",
            "all": "Tüm Tespitler"
        }
        self.current_mode = "face_eyes"  # Başlangıç modu
        
        # Renk paletleri (BGR formatında)
        self.colors = {
            "face": (0, 255, 0),      # Yeşil - Yüz
            "eye": (255, 0, 0),       # Mavi - Göz
            "smile": (0, 255, 255),   # Sarı - Gülümseme
            "profile": (0, 165, 255), # Turuncu - Profil
            "text": (255, 255, 0)     # Turkuaz - Yazı
        }
        
        # İstatistikler
        self.face_count_history = []  # Yüz sayısı geçmişi
        self.detection_times = []     # Tespit süreleri
        self.max_history = 50         # Geçmişte tutulacak maksimum değer
        
        # FPS hesaplama
        self.prev_time = 0
        self.fps = 0
        
        # Ek özellikler
        self.show_landmarks = True    # Yüz işaret noktaları
        self.show_info = True         # Bilgi paneli
        self.show_histogram = False   # Histogram gösterimi
        self.face_zoom = False        # Yüz yakınlaştırma
        self.emotion_detection = False # Temel duygu tespiti
        
        print("\nSistem başarıyla başlatıldı!")
        print("KOMUTLAR:")
        print("  SPACE: Mod değiştir")
        print("  l: İşaret noktalarını aç/kapat")
        print("  i: Bilgi panelini aç/kapat")
        print("  h: Histogramı aç/kapat")
        print("  z: Yüz yakınlaştırmayı aç/kapat")
        print("  e: Duygu tespitini aç/kapat")
        print("  s: Ekran görüntüsü al")
        print("  q: Çıkış")
        print("="*50)
    
    def calculate_fps(self):
        """FPS (Saniyedeki Kare Sayısı) hesaplar"""
        current_time = time.time()
        time_diff = current_time - self.prev_time
        
        if time_diff > 0:
            self.fps = 1 / time_diff
        
        self.prev_time = current_time
        return self.fps
    
    def detect_faces(self, gray_frame, color_frame, scale=1.1, neighbors=5):
        """
        Yüzleri tespit eder
        """
        faces = []
        
        # Ön yüz tespiti
        detected_faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=scale,      # Her ölçekte %10 büyütme
            minNeighbors=neighbors, # Minimum komşu sayısı
            minSize=(30, 30)        # Minimum yüz boyutu
        )
        
        for (x, y, w, h) in detected_faces:
            # Yüz ROI'si (Region of Interest)
            face_roi_gray = gray_frame[y:y+h, x:x+w]
            face_roi_color = color_frame[y:y+h, x:x+w]
            
            # Yüz merkezi
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Yüz özellikleri
            face_features = {
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'roi_gray': face_roi_gray,
                'roi_color': face_roi_color,
                'eyes': [],
                'smile': None,
                'confidence': 1.0
            }
            
            faces.append(face_features)
        
        return faces
    
    def detect_eyes(self, face_roi_gray, face_bbox):
        """
        Bir yüz bölgesinde gözleri tespit eder
        """
        eyes = []
        
        # Göz tespiti (sadece yüz bölgesinde)
        detected_eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Gözleri filtrele ve sırala
        for (ex, ey, ew, eh) in detected_eyes:
            # Göz merkezi
            eye_center_x = ex + ew // 2
            eye_center_y = ey + eh // 2
            
            # Sol ve sağ göz ayırımı
            eye_side = "sol" if eye_center_x < face_bbox[2] // 2 else "sag"
            
            eyes.append({
                'bbox': (ex, ey, ew, eh),
                'center': (eye_center_x, eye_center_y),
                'side': eye_side
            })
        
        # Gözleri X koordinatına göre sırala (soldan sağa)
        eyes.sort(key=lambda e: e['center'][0])
        
        return eyes
    
    def detect_smile(self, face_roi_gray):
        """
        Gülümsemeyi tespit eder
        """
        smiles = self.smile_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        if len(smiles) > 0:
            # En büyük gülümseme bölgesini al
            x, y, w, h = max(smiles, key=lambda s: s[2] * s[3])
            return (x, y, w, h)
        
        return None
    
    def detect_profile_faces(self, gray_frame):
        """
        Profil yüzleri tespit eder
        """
        profiles = self.profile_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return profiles
    
    def estimate_emotion(self, face_roi_gray, eyes):
        """
        Temel duygu tahmini yapar
        Göz ve ağız pozisyonlarına göre
        """
        if len(eyes) < 2:  # İki göz görünmüyorsa
            return "Belirsiz"
        
        # Basit duygu tespiti için
        eye_distance = abs(eyes[0]['center'][0] - eyes[1]['center'][0])
        avg_eye_height = (eyes[0]['bbox'][3] + eyes[1]['bbox'][3]) / 2
        
        # ROI histogram analizi
        hist = cv2.calcHist([face_roi_gray], [0], None, [256], [0, 256])
        brightness = np.mean(face_roi_gray)
        
        # Basit kurallara dayalı duygu tahmini
        if brightness > 150 and eye_distance > avg_eye_height * 1.5:
            return "Mutlu"
        elif brightness < 100 and eye_distance < avg_eye_height * 1.2:
            return "Uzgun"
        elif eye_distance > avg_eye_height * 1.8:
            return "Saskin"
        else:
            return "Nötr"
    
    def draw_face_landmarks(self, frame, face):
        """
        Yüz işaret noktalarını çizer
        """
        x, y, w, h = face['bbox']
        
        # Yüz merkezi
        center_x, center_y = face['center']
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
        
        # Yüz köşe noktaları
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)           # Sol üst
        cv2.circle(frame, (x+w, y), 3, (255, 0, 0), -1)         # Sağ üst
        cv2.circle(frame, (x, y+h), 3, (255, 0, 0), -1)         # Sol alt
        cv2.circle(frame, (x+w, y+h), 3, (255, 0, 0), -1)       # Sağ alt
        
        # Yüz çizgileri
        # Göz hattı
        eye_line_y = y + h // 3
        cv2.line(frame, (x, eye_line_y), (x+w, eye_line_y), (255, 255, 0), 1)
        
        # Burun hattı
        nose_line_y = y + h // 2
        cv2.line(frame, (x, nose_line_y), (x+w, nose_line_y), (255, 255, 0), 1)
        
        # Ağız hattı
        mouth_line_y = y + 2*h // 3
        cv2.line(frame, (x, mouth_line_y), (x+w, mouth_line_y), (255, 255, 0), 1)
        
        # Dikey çizgiler
        vertical_lines = [x + w//4, x + w//2, x + 3*w//4]
        for vx in vertical_lines:
            cv2.line(frame, (vx, y), (vx, y+h), (255, 255, 0), 1)
    
    def draw_face_info(self, frame, face, index):
        """
        Yüz bilgilerini çizer
        """
        x, y, w, h = face['bbox']
        
        # Yüz numarası
        cv2.putText(frame, f"Yuz {index+1}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 2)
        
        # Boyut bilgisi
        cv2.putText(frame, f"{w}x{h}", (x, y+h+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)
        
        # Koordinat bilgisi
        cv2.putText(frame, f"({face['center'][0]},{face['center'][1]})", 
                   (x, y+h+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)
        
        # Göz sayısı
        if len(face['eyes']) > 0:
            eye_text = f"Goz: {len(face['eyes'])}"
            cv2.putText(frame, eye_text, (x, y+h+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)
        
        # Duygu tespiti
        if self.emotion_detection and len(face['eyes']) >= 2:
            emotion = self.estimate_emotion(face['roi_gray'], face['eyes'])
            cv2.putText(frame, f"Duygu: {emotion}", (x, y+h+65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def draw_detection_info(self, frame, face_count, detection_time):
        """
        Genel tespit bilgilerini çizer
        """
        if not self.show_info:
            return
        
        info_y = 30
        line_height = 25
        
        # Arka plan dikdörtgeni
        cv2.rectangle(frame, (10, 10), (300, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 180), (255, 255, 255), 1)
        
        # Başlık
        cv2.putText(frame, "Yuz ve Goz Tespit Sistemi", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text"], 2)
        
        # Mod bilgisi
        cv2.putText(frame, f"Mod: {self.detection_modes[self.current_mode]}", 
                   (20, info_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Yüz sayısı
        cv2.putText(frame, f"Tespit Edilen Yuz: {face_count}", 
                   (20, info_y + line_height*2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (20, info_y + line_height*3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Tespit süresi
        cv2.putText(frame, f"Tespit Suresi: {detection_time*1000:.1f} ms", 
                   (20, info_y + line_height*4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Ek özellikler durumu
        features_text = "Ozellikler: "
        if self.show_landmarks:
            features_text += "✓Isaret "
        if self.emotion_detection:
            features_text += "✓Duygu "
        if self.face_zoom:
            features_text += "✓Zoom "
        
        cv2.putText(frame, features_text, (20, info_y + line_height*5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Kılavuz
        cv2.putText(frame, "SPACE: Mod, l: Isaret, i: Bilgi", 
                   (20, info_y + line_height*6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def create_histogram_display(self, gray_frame, face_rois):
        """
        Histogram görüntüsü oluşturur
        """
        hist_height = 100
        hist_width = 300
        hist_display = np.zeros((hist_height, hist_width), dtype=np.uint8)
        
        if len(face_rois) > 0:
            # İlk yüzün histogramını al
            face_roi = face_rois[0]['roi_gray']
            
            # Histogramı hesapla
            hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
            
            # Histogramı çiz
            for i in range(255):
                cv2.line(hist_display, 
                        (i, hist_height - int(hist[i])),
                        (i, hist_height),
                        (255, 255, 255), 1)
        
        return hist_display
    
    def create_face_zoom_display(self, face_roi_color, original_size):
        """
        Yüz yakınlaştırma görüntüsü oluşturur
        """
        if face_roi_color is None or face_roi_color.size == 0:
            return None
        
        # Yüz ROI'sini büyüt
        zoom_factor = 3  # 3 kat büyüt
        zoomed_face = cv2.resize(face_roi_color, 
                                (face_roi_color.shape[1] * zoom_factor, 
                                 face_roi_color.shape[0] * zoom_factor))
        
        # Kenarlık ekle
        zoomed_face = cv2.copyMakeBorder(zoomed_face, 10, 10, 10, 10,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # Bilgi ekle
        cv2.putText(zoomed_face, "YAKINLASTIRILMIS YUZ", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return zoomed_face
    
    def run_detection(self):
        """
        Ana tespit döngüsü
        """
        screenshot_count = 0
        
        while True:
            # Frame yakalama başlangıç zamanı
            start_time = time.time()
            
            # Kameradan frame al
            ret, frame = self.cap.read()
            if not ret:
                print("Kamera okunamadı!")
                break
            
            # Aynalı görüntü
            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()
            
            # Gri ton dönüşümü
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Histogram eşitleme (kontrast artırma)
            gray_eq = cv2.equalizeHist(gray)
            
            # FPS hesapla
            self.calculate_fps()
            
            # TESPİTLER
            faces = []
            profile_faces = []
            
            # Moda göre tespit yap
            if self.current_mode in ["face_only", "face_eyes", "face_smile", "all"]:
                faces = self.detect_faces(gray_eq, frame)
            
            if self.current_mode in ["profile", "all"]:
                profile_faces = self.detect_profile_faces(gray_eq)
            
            # Her yüz için ek tespitler
            for face in faces:
                # Göz tespiti
                if self.current_mode in ["face_eyes", "all"]:
                    face['eyes'] = self.detect_eyes(face['roi_gray'], face['bbox'])
                
                # Gülümseme tespiti
                if self.current_mode in ["face_smile", "all"]:
                    face['smile'] = self.detect_smile(face['roi_gray'])
            
            # Tespit süresini hesapla
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            if len(self.detection_times) > self.max_history:
                self.detection_times.pop(0)
            
            # Yüz sayısını kaydet
            self.face_count_history.append(len(faces))
            if len(self.face_count_history) > self.max_history:
                self.face_count_history.pop(0)
            
            # ÇİZİMLER
            # 1. Yüzleri çiz
            for i, face in enumerate(faces):
                x, y, w, h = face['bbox']
                
                # Yüz dikdörtgeni
                cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors["face"], 2)
                
                # Yüz bilgileri
                self.draw_face_info(frame, face, i)
                
                # İşaret noktaları
                if self.show_landmarks:
                    self.draw_face_landmarks(frame, face)
                
                # 2. Gözleri çiz
                for eye in face['eyes']:
                    ex, ey, ew, eh = eye['bbox']
                    # Göz koordinatlarını tam frame'e dönüştür
                    abs_ex = x + ex
                    abs_ey = y + ey
                    
                    # Göz dikdörtgeni
                    cv2.rectangle(frame, (abs_ex, abs_ey), 
                                 (abs_ex+ew, abs_ey+eh), self.colors["eye"], 1)
                    
                    # Göz merkezi
                    cv2.circle(frame, (abs_ex + ew//2, abs_ey + eh//2), 
                              2, (255, 255, 0), -1)
                    
                    # Göz tarafı yazısı
                    cv2.putText(frame, eye['side'][0].upper(), 
                               (abs_ex, abs_ey-5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, self.colors["eye"], 1)
                
                # 3. Gülümsemeyi çiz
                if face['smile'] is not None:
                    sx, sy, sw, sh = face['smile']
                    # Gülümseme koordinatlarını tam frame'e dönüştür
                    abs_sx = x + sx
                    abs_sy = y + sy
                    
                    cv2.rectangle(frame, (abs_sx, abs_sy), 
                                 (abs_sx+sw, abs_sy+sh), self.colors["smile"], 1)
                    cv2.putText(frame, "Gulumseme", (abs_sx, abs_sy-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["smile"], 1)
            
            # 4. Profil yüzleri çiz
            for (x, y, w, h) in profile_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors["profile"], 2)
                cv2.putText(frame, "Profil Yuz", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["profile"], 2)
            
            # 5. Bilgi paneli
            self.draw_detection_info(frame, len(faces), detection_time)
            
            # 6. Histogram gösterimi
            if self.show_histogram and len(faces) > 0:
                hist_display = self.create_histogram_display(gray_eq, faces)
                if hist_display is not None:
                    # Histogramı frame'in sağ üstüne yerleştir
                    hist_x = frame.shape[1] - hist_display.shape[1] - 10
                    hist_y = 10
                    frame[hist_y:hist_y+hist_display.shape[0], 
                          hist_x:hist_x+hist_display.shape[1]] = \
                        cv2.cvtColor(hist_display, cv2.COLOR_GRAY2BGR)
                    
                    # Histogram başlığı
                    cv2.putText(frame, "Yuz Histogrami", 
                               (hist_x, hist_y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, (255, 255, 255), 1)
            
            # 7. Yüz yakınlaştırma
            if self.face_zoom and len(faces) > 0:
                zoomed_face = self.create_face_zoom_display(faces[0]['roi_color'], 
                                                           frame.shape)
                if zoomed_face is not None:
                    # Zoom'u frame'in sağ altına yerleştir
                    zoom_x = frame.shape[1] - zoomed_face.shape[1] - 10
                    zoom_y = frame.shape[0] - zoomed_face.shape[0] - 10
                    
                    # Zoom boyutu frame'i aşmıyorsa
                    if zoom_x > 0 and zoom_y > 0:
                        frame[zoom_y:zoom_y+zoomed_face.shape[0], 
                              zoom_x:zoom_x+zoomed_face.shape[1]] = zoomed_face
            
            # Ana pencereyi göster
            cv2.imshow('Yuz ve Goz Tespit Sistemi', frame)
            
            # Ek pencereler
            if self.show_histogram and len(faces) > 0:
                # Gri tonlu görüntü
                cv2.imshow('Gri Tonlu Goruntu', gray_eq)
                
                # İlk yüzün ROI'si
                if len(faces) > 0:
                    cv2.imshow('Ilk Yuz ROI', faces[0]['roi_color'])
            
            # TUŞ KONTROLLERİ
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Çıkış
                break
            elif key == ord(' '):  # Space: Mod değiştir
                modes = list(self.detection_modes.keys())
                current_idx = modes.index(self.current_mode)
                self.current_mode = modes[(current_idx + 1) % len(modes)]
                print(f"Mod degistirildi: {self.detection_modes[self.current_mode]}")
            elif key == ord('l'):  # İşaret noktaları
                self.show_landmarks = not self.show_landmarks
                status = "acik" if self.show_landmarks else "kapali"
                print(f"Isaret noktalari: {status}")
            elif key == ord('i'):  # Bilgi paneli
                self.show_info = not self.show_info
                status = "acik" if self.show_info else "kapali"
                print(f"Bilgi paneli: {status}")
            elif key == ord('h'):  # Histogram
                self.show_histogram = not self.show_histogram
                status = "acik" if self.show_histogram else "kapali"
                print(f"Histogram: {status}")
            elif key == ord('z'):  # Yüz yakınlaştırma
                self.face_zoom = not self.face_zoom
                status = "acik" if self.face_zoom else "kapali"
                print(f"Yuz yakınlastirma: {status}")
            elif key == ord('e'):  # Duygu tespiti
                self.emotion_detection = not self.emotion_detection
                status = "acik" if self.emotion_detection else "kapali"
                print(f"Duygu tespiti: {status}")
            elif key == ord('s'):  # Ekran görüntüsü
                screenshot_count += 1
                filename = f"yuz_tespit_{screenshot_count}.jpg"
                cv2.imwrite(filename, original_frame)
                print(f"Ekran goruntusu kaydedildi: {filename}")
        
        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()
        
        # İstatistikleri göster
        self.show_statistics()
    
    def show_statistics(self):
        """
        Tespit istatistiklerini göster
        """
        if len(self.face_count_history) == 0:
            print("Istatistik yok!")
            return
        
        print("\n" + "="*50)
        print("TESPI ISTATISTIKLERI")
        print("="*50)
        
        # Ortalama yüz sayısı
        avg_faces = np.mean(self.face_count_history)
        max_faces = np.max(self.face_count_history)
        print(f"Ortalama tespit edilen yuz: {avg_faces:.2f}")
        print(f"Maksimum tespit edilen yuz: {max_faces}")
        
        # Ortalama tespit süresi
        if self.detection_times:
            avg_time = np.mean(self.detection_times) * 1000  # ms cinsinden
            print(f"Ortalama tespit suresi: {avg_time:.2f} ms")
        
        # Histogram gösterimi
        plt.figure(figsize=(12, 5))
        
        # Yüz sayısı geçmişi
        plt.subplot(1, 2, 1)
        plt.plot(self.face_count_history, 'b-', linewidth=2)
        plt.title('Yuz Sayisi Gecmisi')
        plt.xlabel('Frame')
        plt.ylabel('Yuz Sayisi')
        plt.grid(True, alpha=0.3)
        
        # Tespit süreleri
        if self.detection_times:
            plt.subplot(1, 2, 2)
            times_ms = [t * 1000 for t in self.detection_times]  # ms'ye çevir
            plt.plot(times_ms, 'r-', linewidth=2)
            plt.title('Tespit Sureleri')
            plt.xlabel('Frame')
            plt.ylabel('Sure (ms)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# BASİT YÜZ TESPİTİ (Alternatif)
def simple_face_detection():
    """
    Basit yüz tespiti - sadece temel fonksiyonlar
    """
    # Cascade'leri yükle
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    
    if face_cascade.empty() or eye_cascade.empty():
        print("Cascade dosyalari yuklenemedi!")
        return
    
    # Kamerayı aç
    cap = cv2.VideoCapture(0)
    
    print("BASIT YUZ TESPITI")
    print("q: Cikis")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Ayna efekti
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüz tespiti
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Yüz çerçevesi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Yuz", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Yüz ROI'si
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Göz tespiti
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                cv2.putText(roi_color, "Goz", (ex, ey-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Yüz bilgisi
            info_text = f"Boyut: {w}x{h}"
            cv2.putText(frame, info_text, (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Yüz sayısı
        cv2.putText(frame, f"Yuz Sayisi: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Göster
        cv2.imshow('Basit Yuz Tespiti', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# REALTIME DUYGU ANALİZİ
def realtime_emotion_analysis():
    """
    Gerçek zamanlı duygu analizi
    """
    # Cascade'leri yükle
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(0)
    
    # Duygu renkleri
    emotion_colors = {
        "Mutlu": (0, 255, 0),      # Yeşil
        "Uzgun": (255, 0, 0),      # Mavi
        "Kizgin": (0, 0, 255),     # Kırmızı
        "Saskin": (0, 255, 255),   # Sarı
        "Normal": (255, 255, 0)    # Turkuaz
    }
    
    print("REALTIME DUYGU ANALIZI")
    print("q: Cikis")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüz tespiti
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Yüz ROI'si
            face_roi = gray[y:y+h, x:x+w]
            
            # Basit duygu analizi (gerçek bir model yerine basit kurallar)
            emotion = "Normal"
            color = emotion_colors[emotion]
            
            # Histogram analizi
            brightness = np.mean(face_roi)
            
            # Basit kurallar
            if brightness > 180:
                emotion = "Mutlu"
            elif brightness < 80:
                emotion = "Uzgun"
            elif w/h > 1.3:  # Geniş yüz
                emotion = "Saskin"
            
            color = emotion_colors.get(emotion, (255, 255, 255))
            
            # Yüz çerçevesi (duyguya göre renk)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Duygu bilgisi
            cv2.putText(frame, emotion, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Duygu şiddeti (parlaklık bazlı)
            intensity = min(100, abs(brightness - 128) * 2) / 100
            cv2.putText(frame, f"Siddet: %{intensity*100:.0f}", (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Duygu dağılımı
        emotion_counts = {"Mutlu": 0, "Uzgun": 0, "Normal": 0, "Saskin": 0}
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            brightness = np.mean(face_roi)
            
            if brightness > 180:
                emotion_counts["Mutlu"] += 1
            elif brightness < 80:
                emotion_counts["Uzgun"] += 1
            elif w/h > 1.3:
                emotion_counts["Saskin"] += 1
            else:
                emotion_counts["Normal"] += 1
        
        # Bilgi paneli
        y_offset = 30
        for emotion, count in emotion_counts.items():
            if count > 0:
                color = emotion_colors[emotion]
                cv2.putText(frame, f"{emotion}: {count}", 
                           (frame.shape[1] - 150, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
        
        cv2.imshow('Realtime Duygu Analizi', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ANA PROGRAM
if __name__ == "__main__":
    print("YÜZ VE GÖZ TESPİT SİSTEMİ")
    print("="*50)
    print("1. Gelişmiş Yüz ve Göz Tespiti")
    print("2. Basit Yüz Tespiti")
    print("3. Realtime Duygu Analizi")
    print("4. Fotoğraftan Yüz Tespiti")
    
    choice = input("\nSeçiminiz (1-4): ")
    
    if choice == "1":
        detector = FaceEyeDetector()
        detector.run_detection()
    elif choice == "2":
        simple_face_detection()
    elif choice == "3":
        realtime_emotion_analysis()
    elif choice == "4":
        # Fotoğraftan yüz tespiti
        image_path = input("Fotoğraf dosya yolu: ")
        
        # Cascade'leri yükle
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Fotoğrafı yükle
        image = cv2.imread(image_path)
        if image is None:
            print("Fotoğraf yüklenemedi!")
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Yüz tespiti
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            print(f"Tespit edilen yüz sayısı: {len(faces)}")
            
            for (x, y, w, h) in faces:
                # Yüz çerçevesi
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Göz tespiti
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 1)
            
            # Sonucu göster
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Yüz Tespiti - {len(faces)} yüz bulundu')
            plt.axis('off')
            plt.show()
            
            # Kaydet
            save_choice = input("Sonucu kaydetmek ister misiniz? (e/h): ")
            if save_choice.lower() == 'e':
                output_path = "yuz_tespit_sonucu.jpg"
                cv2.imwrite(output_path, image)
                print(f"Sonuç kaydedildi: {output_path}")
    else:
        print("Geçersiz seçim!")
        
        
        
        """
        1. TEMEL ALGORİTMA: HAAR CASCADE CLASSIFIER
        Bu kodun merkezinde Viola-Jones algoritmasına dayanan Haar Cascade sınıflandırıcılar kullanılmıştır:

        Viola-Jones Algoritması (2001)
        python
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        Çalışma Prensibi:

        Haar-like Features (Haar Benzeri Özellikler):

        Görüntüdeki piksel gruplarının parlaklık farklarını ölçer

        Temel desenler: kenarlar, çizgiler, merkez bölgeler

        Integral Image (Birleşik Görüntü):

        Hızlı özellik hesaplama için

        Herhangi bir dikdörtgenin piksel toplamını O(1) zamanında hesaplar

        AdaBoost (Adaptive Boosting):

        Zayıf sınıflandırıcıları birleştirerek güçlü sınıflandırıcı oluşturur

        En iyi özellikleri seçer

        Cascade Structure (Kademeli Yapı):

        Hızlı elenme mekanizması

        Yüz olmayan bölgeleri erken aşamada eleme
        """