import cv2                 # OpenCV
import time                # Per timing e FPS
from collections import deque  # Per lo smoothing
import mediapipe as mp     # MediaPipe
from threading import Thread # Per la soluzione "Pro"

# ---------- Configurazione Ottimizzata ----------
# Impostazioni che hanno funzionato per te:
WIDTH, HEIGHT, REQ_FPS = 720, 640, 30
FOURCC = "YUYV"                         # 'YUYV' ha sbloccato i 15 FPS

MODEL_COMPLEXITY = 0
DRAW_LANDMARKS = True
PERFORMANCE_MODE = False                # Non più necessario con il threading

MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5

SMOOTH_N   = 3
DEBOUNCE_N = 2

FPS_UPDATE_INTERVAL = 30                # Aggiorna display FPS ogni N frame

# Impostazioni Esposizione (meglio 'False' per webcam integrate)
MANUAL_EXPOSURE = False
EXPOSURE_VALUE = -6

# Impostazioni display
FONT_SCALE = 0.7
FONT_THICKNESS = 2
LINE_TYPE = cv2.LINE_AA

# ----------------------------
# CLASSE "PRO" PER LA WEBCAM
# ----------------------------
class WebcamStream:
    """
    Classe per leggere la webcam in un thread dedicato,
    per evitare che cv2.read() blocchi il thread principale.
    """
    def __init__(self, src=0, width=640, height=480, fps=30, fourcc_str="YUYV"):
        print(f"[WebcamStream] Avvio stream: {width}x{height} @ {fps} FPS con {fourcc_str}")
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        
        # Applica le impostazioni richieste
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
        
        # Impostazioni di performance
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if MANUAL_EXPOSURE:
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.stream.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VALUE)
        else:
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # Modalità automatica
            
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.stream.set(cv2.CAP_PROP_AUTO_WB, 0)
        
        # Legge il primo frame
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            print("[WebcamStream] ERRORE: Impossibile leggere il primo frame.")

        # Flag per terminare il thread
        self.stopped = False
        print("[WebcamStream] Stream inizializzato.")

    def start(self):
        # Avvia il thread per leggere i frame
        t = Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        # Loop finché non viene fermato
        while True:
            if self.stopped:
                return
            # Legge il prossimo frame dalla webcam
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                # Se la camera si disconnette, ferma il thread
                print("[WebcamStream] Camera disconnessa?")
                self.stopped = True

    def read(self):
        # Ritorna il frame più recente
        return self.frame

    def stop(self):
        # Ferma il thread e rilascia la risorsa
        print("[WebcamStream] Fermo lo stream...")
        self.stopped = True
        time.sleep(0.1) # Dà al thread il tempo di terminare
        self.stream.release()
        print("[WebcamStream] Rilasciato.")

# ----------------------------
# FUNZIONI DELLO SCRIPT (Invariate)
# ----------------------------

def count_fingers(landmarks, handedness, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = 0
    
    for tip, pip in zip(tips, pips):
        if pts[tip][1] < pts[pip][1]:
            fingers += 1
    
    thumb_tip, thumb_ip = 4, 3
    if handedness == "Right":
        if pts[thumb_tip][0] < pts[thumb_ip][0]:
            fingers += 1
    else:
        if pts[thumb_tip][0] > pts[thumb_ip][0]:
            fingers += 1
    return min(fingers, 5)

def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    )
    return mp_hands, mp_draw, hands

def update_hand_state(state, label, fingers_raw):
    # Gestisce il caso 'None' quando una mano scompare
    if fingers_raw is None:
        if state[label]["shown"] is not None:
            # Resetta lo stato se la mano è scomparsa
            state[label] = {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0}
        return

    st = state[label]
    if st["shown"] is None:
        st["shown"] = fingers_raw
    st["buf"].append(fingers_raw)
    maj = max(set(st["buf"]), key=st["buf"].count)
    if (st["maj"] is None) or (maj != st["maj"]):
        st["maj"] = maj
        st["stable"] = 1
    else:
        st["stable"] += 1
    if (maj != st["shown"]) and (st["stable"] >= DEBOUNCE_N):
        st["shown"] = maj

def draw_info_overlay(frame, state, fps, h, w):
    left_val = state["Left"]["shown"] if state["Left"]["shown"] is not None else "-"
    right_val = state["Right"]["shown"] if state["Right"]["shown"] is not None else "-"
    
    x_offset = 10
    y_start = 25
    y_spacing = 25
    
    cv2.putText(frame, f"Left: {left_val}", (x_offset, y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS, LINE_TYPE)
    cv2.putText(frame, f"Right: {right_val}", (x_offset, y_start + y_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS, LINE_TYPE)
    
    # Questo ora misura l'FPS di elaborazione, non della camera
    cv2.putText(frame, f"Proc FPS: {fps:.1f}", (x_offset, y_start + 2 * y_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE - 0.1, (0, 255, 0), FONT_THICKNESS, LINE_TYPE)
    
    cv2.putText(frame, f"{w}x{h}", (x_offset, h - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE - 0.2, (128, 128, 128), 1, LINE_TYPE)

# ----------------------------
# FUNZIONE MAIN (Modificata)
# ----------------------------
def main():
    cv2.setNumThreads(1)
    
    # 1. AVVIA LA WEBCAM NEL SUO THREAD
    # Non usiamo più setup_camera(), ma la nostra nuova classe
    vs = WebcamStream(
        src=0, 
        width=WIDTH, 
        height=HEIGHT, 
        fps=REQ_FPS, 
        fourcc_str=FOURCC
    ).start()
    
    # Dà alla camera 1 secondo per "scaldarsi" e stabilizzarsi
    print("[Main] Attendo 1s che la camera si stabilizzi...")
    time.sleep(1.0)
    print("[Main] Avvio elaborazione.")

    # Inizializza MediaPipe
    mp_hands, mp_draw, hands = initialize_mediapipe()
    
    # Stato (invariato)
    state = {
        "Left":  {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0},
        "Right": {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0},
    }
    
    # Tracking performance (invariato)
    last_fps_update = time.time()
    fps_frame_count = 0
    fps = 0.0
    
    # Landmark e risultati (invariati)
    draw_landmarks = DRAW_LANDMARKS
    last_results = None
    
    try:
        while True:
            # 2. LEGGE IL FRAME DAL THREAD (non bloccante)
            # Sostituisce 'ok, frame = cap.read()'
            frame = vs.read()
            
            # Se il frame non è ancora pronto, salta il giro
            if frame is None:
                continue
                
            frame = cv2.flip(frame, 1)  # Specchia
            h, w = frame.shape[:2]
            
            fps_frame_count += 1
            
            # 3. PROCESSIAMO OGNI FRAME!
            # Rimuoviamo la logica 'PROCESS_EVERY_N_FRAMES'
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False # Ottimizzazione: segna come non scrivibile
            results = hands.process(rgb)
            rgb.flags.writeable = True # Riabilita la scrittura
            last_results = results
            
            seen = {"Left": False, "Right": False}
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    if label not in state:
                        state[label] = {"buf": deque(maxlen=SMOOTH_N), "shown": None, "maj": None, "stable": 0}
                    
                    seen[label] = True
                    fingers_raw = count_fingers(landmarks.landmark, label, w, h)
                    update_hand_state(state, label, fingers_raw)
            
            # Resetta le mani che non sono più visibili
            for label in ["Left", "Right"]:
                if not seen[label]:
                    update_hand_state(state, label, None) # Invia 'None' per resettare
            
            # Disegna i landmark dall'ultimo risultato (invariato)
            if draw_landmarks and last_results and last_results.multi_hand_landmarks:
                for landmarks in last_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Calcola FPS (ora misura l'FPS di elaborazione)
            if fps_frame_count >= FPS_UPDATE_INTERVAL:
                now = time.time()
                elapsed = now - last_fps_update
                if elapsed > 0:
                    fps = fps_frame_count / elapsed
                last_fps_update = now
                fps_frame_count = 0
            
            # Disegna overlay
            draw_info_overlay(frame, state, fps, h, w)
            
            # Mostra frame
            cv2.imshow(f"Threaded Finger Counter - {WIDTH}x{HEIGHT}", frame)
            
            # Input (invariato)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('l'):
                draw_landmarks = not draw_landmarks
                print(f"Landmarks: {'ON' if draw_landmarks else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # 4. FERMA IL THREAD DELLA WEBCAM
        # Sostituisce 'cap.release()'
        vs.stop()
        cv2.destroyAllWindows()
        print("Cleanup completo. Uscita.")

if __name__ == "__main__":
    main()