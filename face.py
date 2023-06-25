# Подключение библиотеки компьютерного зрения
import cv2
# Подключение библиотеки для работы с аргументами при вызове
import argparse

# Подключаем парсер аргументов командной строки
parser = argparse.ArgumentParser()
# Добавляем аргумент для работы с изображениями
parser.add_argument('--image')
# Сохраняем аргументы в отдельную переменную
args = parser.parse_args()

# Функция определения лиц
def highlightFace(net, frame, conf_threshold = 0.7):
    # Делаем копию текущего кадра
    frameOpencvDnn = frame.copy()
    # Высота и ширина кадра
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # Преобразуем картинку в двоичный пиксельный объект
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    # Устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)
    # Выполняем прямой проход для распознавания лиц
    detections = net.forward()
    # Переменная для рамок вокруг лица
    faceBoxes = []
    # Перебираем все блоки после распознавания
    for i in range(detections.shape[2]):
        # Получаем результат вычислений для очередного элемента
        confidence = detections[0, 0, i, 2]
        # Если результат превышает порог срабатывания — это лицо
        if confidence > conf_threshold:
            # Формируем координаты рамки
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            # Добавляем их в общую переменную
            faceBoxes.append([x1, y1, x2, y2])
            # Рисуем рамку на кадре
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    # Возвращаем кадр с рамками
    return frameOpencvDnn, faceBoxes


# Загружаем веса для распознования лиц
faceProto = "protos/opencv_face_detector.pbtxt"
# Загружаем конфигурацию нейросети распознования лиц(слои и связи нейронов)
faceModel = "models/opencv_face_detector_uint8.pb"
# Загружаем веса для определения пола
genderProto = "protos/gender_deploy.prototxt"
# Загружаем конфигурацию нейросети определения пола
genderModel = "models/gender_net.caffemodel"
# Загружаем веса для определения возраста
ageProto = "protos/age_deploy.prototxt"
# Загружаем конфигурацию нейросети определения возраста
ageModel = "models/age_net.caffemodel"

# Настраиваем свет
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Итоговые результаты работы нейросетей для пола и возраста
genderList = ['Male ', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Запускаем нейросеть по распознованию лиц
faceNet = cv2.dnn.readNet(faceModel, faceProto)
# Запускаем нейросети по определению пола и возраста
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Получаем видео с камеры, если был указан аргумент с картинкой — берём картинку как источник
video = cv2.VideoCapture(args.image if args.image else 0)
# Пока не нажата любая клавиша — выполняем цикл
while cv2.waitKey(1) < 0:
    # Получаем очередной кадр с камеры
    hasFrame, frame = video.read()
    # Если кадра нет
    if not hasFrame:
        # Останавливаемся и выходим из цикла
        cv2.waitKey()
        break
    # Распознаём лица в кадре
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    # Если лиц нет
    if not faceBoxes:
        print("Лица не распознаны.")
    # Перебираем все найденные лица в кадре
    for faceBox in faceBoxes:
        # Получаем изображение лица на основе рамки
        face = frame[max(0, faceBox[1]):
                     min(faceBox[3], frame.shape[0] - 1), max(0, faceBox[0])
                     :min(faceBox[2], frame.shape[1] - 1)]
        # Получаем на этой основе новый бинарный пиксельный объект
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        # Оотправляем его в нейросеть для определения пола
        genderNet.setInput(blob)
        # Получаем результат работы нейросети
        genderPreds = genderNet.forward()
        # Выбираем пол на основе этого результата
        gender = genderList[genderPreds[0].argmax()]
        # Отправляем результат в переменную с полом
        print('Gender: ', gender)

        # Делаем то же самое для возраста
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print('Age:', age[1:-1],' years')

        # Добавляем текст возле каждой рамки в кадре
        cv2.putText(resultImg, gender + '' + age, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        # Выводим итоговую картинку
        cv2.imshow("Detecting age and gender", resultImg)