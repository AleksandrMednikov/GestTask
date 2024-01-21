import torchvision.models as models
import torch
import torch.nn as nn
import cv2
import numpy as np
import mediapipe as mp

## Настройки
path_model = 'vgg19_model_mediapipe.pth'
face_cascade = cv2.CascadeClassifier('D:/Workplace/pythonprojects/webGest/pythonProject/haarcascade_frontalface_alt.xml')
step = 2  # делает предикт каждый указанный по счету кадр для повышения производительности
text = base_text = 'Status :'
x_bias = 25
y_bias = 25
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
last_find_hand = 0

label_keys = {0: '02_l',
             1: '10_down',
             2: '03_fist',
             3: '08_palm_moved',
             4: '07_ok',
             5: '01_palm',
             6: '04_fist_moved',
             7: '09_c',
             8: '05_thumb',
             9: '06_index'}

# Модель
gest_model = models.vgg19(pretrained=True)
num_classes = 10
gest_model.classifier[6] = nn.Linear(4096, num_classes)
gest_model.load_state_dict(torch.load(path_model))


#предикт
def predict_image(model, image):
    image = torch.from_numpy(image).permute(0, 3, 1, 2).float()
    output = model(image)
    _, predicted_class = torch.max(output, 1)
    return label_keys[predicted_class.item()]


def crop_hand(image, x, y, w, h, x_increase=x_bias, y_increase=y_bias):
    image_h, image_w = image.shape[:2]

    x_start = max(0, x - x_increase)
    y_start = max(0, y - y_increase)
    x_end = min(image_w, x + w + x_increase)
    y_end = min(image_h, y + h + y_increase)

    cropped_hand = image[y_start:y_end, x_start:x_end]

    return cropped_hand

cap = cv2.VideoCapture(0)
print('start video flow')
frame_count = 0
while True:
    #захват кадра
    ret, frame = cap.read()
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    frame_height, frame_width, _ = frame.shape

    #детект лица
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=9, minSize=(64, 64))

    for face in faces:
        xF, yF, wF, hF = face
        #face_rect = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (xF, yF), (xF + wF, yF + hF), (186, 175, 201), 2)
        cv2.putText(frame, "You", (xF, yF - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 22, 201), 2)

    # детект руки
    if frame_count % step == 0:
        results = hands.process(cv2.cvtColor(gray3d, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Определение области, содержащей руку
                landmarks_list = [(lm.x * frame_width, lm.y * frame_height) for lm in hand_landmarks.landmark]
                xP, yP, wP, hP = cv2.boundingRect(np.array(landmarks_list).astype(np.int32))

                # Выделение руки в прямоугольник
                hand_cropped = crop_hand(gray3d, xP, yP, wP, hP, x_increase=x_bias, y_increase=y_bias)
                hand_cropped = cv2.resize(hand_cropped, (64, 64))

            x_input = np.array([hand_cropped]) / 255.0

            predict = predict_image(gest_model, x_input)
            text = base_text + predict
            last_find_hand = 1
        else:
            last_find_hand = 0

        frame_count = 0

    if last_find_hand:
        cv2.rectangle(frame, (xP, yP), (xP + wP, yP + hP), (186, 175, 201), 2)
        cv2.putText(frame, text, (xP, yP - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # cv2.putText(frame, text, cordinates, font, font_scale, color, thickness)
    cv2.imshow("flow", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()