import cv2
import time
from ultralytics import YOLO
import mediapipe as mp
from flask import Flask, render_template, Response

# YOLO
model = YOLO('yolov8n.pt')
app= Flask(__name__)

# MediaPipe
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
recording=True


# === Timer trackers ===
face_missing_start = None
head_turn_start = None
phone_detected_start = None

# === Accumulated cheating times ===
face_missing_total = 0
head_turn_total = 0
phone_total = 0

# Duration thresholds (seconds)
ALERT_FACE_MISSING_SEC = 5
DISQUALIFY_DURATION_SEC = 5 * 60

disqualified = False
@app.route('/stop')
def stop():
    global recording
    recording = False
    return "Recording stopped"
def videoCap():
    global recording
    global face_missing_start, head_turn_start, phone_detected_start
    global face_missing_total, head_turn_total, phone_total
    global disqualified

    cap = cv2.VideoCapture(0)

    while recording:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = time.time()
        face_present = False
        head_turn = False

        if results.multi_face_landmarks:
            face_present = True
            for face_landmarks in results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]

                nose_x = nose.x * w
                left_eye_x = left_eye.x * w
                right_eye_x = right_eye.x * w

                eye_center = (left_eye_x + right_eye_x) / 2
                diff = nose_x - eye_center

                if abs(diff) > 30:
                    head_turn = True

        cheating_alerts = []

        ### === FACE MISSING ===
        if not face_present:
            if face_missing_start is None:
                face_missing_start = now
            elapsed = now - face_missing_start
            if elapsed > ALERT_FACE_MISSING_SEC:
                cheating_alerts.append(f"No face > {ALERT_FACE_MISSING_SEC}s")
        else:
            if face_missing_start:
                face_missing_total += now - face_missing_start
                face_missing_start = None

        ### === HEAD TURN ===
        if head_turn:
            if head_turn_start is None:
                head_turn_start = now
            cheating_alerts.append("Head turned")
        else:
            if head_turn_start:
                head_turn_total += now - head_turn_start
                head_turn_start = None

        ### === YOLO ===
        results_yolo = model(frame, verbose=False)
        phone_detected = False
        person_count = 0
        largest_person_area = 0

        for r in results_yolo:
            for box in r.boxes:
                conf = box.conf[0]
                if conf < 0.5:
                    continue
                b = box.xyxy[0]
                c = int(box.cls[0])
                cls_name = model.names[c]
                x1, y1, x2, y2 = map(int, b)
                area = (x2 - x1) * (y2 - y1)

                if cls_name == "person":
                    if area > largest_person_area and area > 10000:
                        largest_person_area = area

                if cls_name == "cell phone":
                    phone_detected = True

        if largest_person_area > 0:
            person_count = 1

        if phone_detected:
            if phone_detected_start is None:
                phone_detected_start = now
            cheating_alerts.append("Phone detected")
        else:
            if phone_detected_start:
                phone_total += now - phone_detected_start
                phone_detected_start = None

        if person_count > 1:
            cheating_alerts.append("Multiple people")

        ### === Add any running time ===
        if face_missing_start:
            face_missing_total_running = now - face_missing_start
        else:
            face_missing_total_running = 0

        if head_turn_start:
            head_turn_total_running = now - head_turn_start
        else:
            head_turn_total_running = 0

        if phone_detected_start:
            phone_total_running = now - phone_detected_start
        else:
            phone_total_running = 0

        ### === Disqualify if any total exceeds ===
        if (face_missing_total + face_missing_total_running >= DISQUALIFY_DURATION_SEC or
            head_turn_total + head_turn_total_running >= DISQUALIFY_DURATION_SEC or
            phone_total + phone_total_running >= DISQUALIFY_DURATION_SEC):
            disqualified = True

        ### === Draw ===
        status = f"Face: {face_present} | Phone: {phone_detected} | People: {person_count} | Head Turn: {head_turn}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        times_status = f"FaceMissing: {face_missing_total + face_missing_total_running:.1f}s | HeadTurn: {head_turn_total + head_turn_total_running:.1f}s | Phone: {phone_total + phone_total_running:.1f}s"
        cv2.putText(frame, times_status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if cheating_alerts:
            alert_text = " | ".join(cheating_alerts)
            cv2.putText(frame, f"CHEATING ALERT: {alert_text}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            print(f"ALERT: {alert_text}")

        if disqualified:
            cv2.putText(frame, "DISQUALIFIED! EXAM TERMINATED", (50, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            print("DISQUALIFIED! Exam stopped.")
            time.sleep(3)
            break

        ret, buffer=cv2.imencode(".jpg",frame)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
    cap.release()

@app.route('/')
def home():
    return render_template("connect.html")
@app.route('/video')
def video():
    return Response(videoCap(),mimetype="multipart/x-mixed-replace; boundary=frame")
if __name__=="__main__":
    app.run(debug=True)
