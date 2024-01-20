import cv2
from keras.models import model_from_json
import numpy as np
from keras_preprocessing.image import load_img

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)



def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Assuming you have already defined the face_cascade, model, etc.
webcam = cv2.VideoCapture(0)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

while True:
    ret, im = webcam.read()
    if not ret:
        break

    labels_and_confidences = []  # To store labels and confidence levels for each detected face

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    try:
        for (p, q, r, s) in faces:
            face_image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (0, 255, 255), 2)
            face_image_resized = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)
            img = extract_features(face_image_resized)
            pred = model.predict(img)
            confidence = np.max(pred)
            prediction_label = labels[pred.argmax()]
            labels_and_confidences.append(f"{prediction_label} ({confidence:.2f})")

            label_position = (p, q)
            cv2.putText(im, f"{prediction_label} ({confidence:.2f})", label_position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)

        # Display the labels and confidences for all detected faces
        all_labels_text = ', '.join(labels_and_confidences)
        cv2.putText(im, all_labels_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        cv2.imshow("Output", im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()
