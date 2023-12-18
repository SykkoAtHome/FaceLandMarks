import cv2
import mediapipe as mp

# Inicjalizacja modułu Mediapipe do wykrywania landmarks na twarzy
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh()

# Wczytaj przykładowy obraz
image_path = "redhead_face_2.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
frame_landmarks = []

results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    for landmarks in results.multi_face_landmarks:
        frame_landmarks.append(landmarks)


mp_drawing.draw_landmarks(image, frame_landmarks[0])
# print(list_landmarks[0])
# Wyświetl wynik
cv2.imshow("Landmarks Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
