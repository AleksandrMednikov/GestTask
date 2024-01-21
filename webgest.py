import torchvision.models as models
import torch
import torch.nn as nn
import cv2
import numpy as np

## Настройки
path_model = 'vgg19_model.pth'
hand_cascade = cv2.CascadeClassifier('D:/Workplace/pythonprojects/webGest/pythonProject/haarcascade_hand (1).xml')
face_cascade = cv2.CascadeClassifier('D:/Workplace/pythonprojects/webGest/pythonProject/haarcascade_frontalface_alt.xml')
step = 2  # делает предикт каждый указанный по счету кадр для повышения производительности
text = base_text = 'Status :'
x_increase = 30
y_increase = 20
last_find_hand = 0
facepalm = 0
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# color = (0, 0, 255)
# cordinates = (7, 465)
# thickness = 2
label_keys = {0: '10_down',
              1: '05_thumb',
              2: '04_fist_moved',
              3: '06_index',
              4: '08_palm_moved',
              5: '07_ok',
              6: '01_palm',
              7: '09_c',
              8: '03_fist',
              9: '02_l'}


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


#обрезка изображения
def crop_hand(image, x, y, w, h, x_increase=x_increase, y_increase=y_increase):
    image_h, image_w = image.shape[:2]

    x_start = max(0, x - x_increase)
    y_start = max(0, y - y_increase)
    x_end = min(image_w, x + w + x_increase)
    y_end = min(image_h, y + h + y_increase)

    cropped_hand = image[y_start:y_end, x_start:x_end]

    return cropped_hand


#поиск пересечений руки и лица
def test_facepalm(face_xwhy, palm_xwhy, frame):
    xf, yf, wf, hf = face_xwhy
    xp, yp, wp, hp = palm_xwhy

    # ones_face = np.ones(cv2.cvtColor(frame[yf:yf + hf, xf:xf + wf], cv2.COLOR_BGR2GRAY).shape)
    # ones_hand_roi = np.ones(cv2.cvtColor(frame[yp:yp + hp, xp:xp + wp], cv2.COLOR_BGR2GRAY).shape)
    zero_frame = np.zeros(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).shape))

    frame_face = zero_frame.copy()
    frame_palm = zero_frame.copy()
    frame_face[yp - y_increase:yp + hp + y_increase, xp - + x_increase:xp + wp + + x_increase] = 1
    frame_palm[yf:yf + hf, xf:xf + wf] = 1
    intersection = np.any(frame_palm.astype(np.uint8) & frame_face.astype(np.uint8))

    return intersection


cap = cv2.VideoCapture(0)
print('start video flow')
frame_count = 0
while True:
    #захват кадра
    ret, frame = cap.read()
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #детект лица
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=9, minSize=(64, 64))

    for face in faces:
        x, y, w, h = face
        #face_rect = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (186, 175, 201), 2)
        cv2.putText(frame, "You", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 22, 201), 2)

    # детект руки
    if frame_count % step == 0:
        facepalm = 0
        hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=2, minSize=(64, 64))

        if len(hands):
            max_size = 0
            top_hands_index = 0
            for index_hand in range(len(hands)):
                x, y, w, h = hands[index_hand]
                hand_roi = crop_hand(frame, x, y, w, h)# вырезаю кусок изображения, соответствующий руке

                # бывает ищет кисть на губах поэтому я так специфически исключил этот вариант
                if len(face):
                    facepalm = test_facepalm(face, hands[index_hand], frame)

                height, width, _ = hand_roi.shape
                if (height + 1 * width + 1) > max_size and not facepalm:
                    hands_index = index_hand

            x, y, w, h = hands[top_hands_index]

            image_h, image_w = frame.shape[:2]
            x_start = max(0, x - x_increase)
            y_start = max(0, y - y_increase)
            x_end = min(image_w, x + w + x_increase)
            y_end = min(image_h, y + h + y_increase)

            # я обучал на трехмерных чернобелых, тут тоже самое
            hands = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)[y_start:y_end, x_start:x_end]

            x_input = np.array([cv2.resize(hands, (64, 64))]) / 255.0

            predict = predict_image(gest_model, x_input)
            text = base_text + predict
            last_find_hand = 1
        else:
            last_find_hand = 0

        frame_count = 0

    if last_find_hand and not facepalm:
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (186, 175, 201), 2)
        cv2.putText(frame, text, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # cv2.putText(frame, text, cordinates, font, font_scale, color, thickness)
    cv2.imshow("flow", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()