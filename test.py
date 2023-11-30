import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras import models
import skimage

model = models.load_model('model_120epoch.h5')

label_dict = {1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: '',
              10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
              18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'}

mp_hand = mp.solutions.hands
hands = mp_hand.Hands()

mp_drawing_utils = mp.solutions.drawing_utils

output_container = st.empty()

# Creamos el objeto SessionState
ss = st.session_state

if 'word' not in ss:
    ss.word = []

img_file_buffer = st.camera_input("test")

if img_file_buffer:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    results = hands.process(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    h, w, c = cv2_img.shape
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h

    hand_landmarks = results.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h

            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

    extra = 30

    x_max_new = min(x_max + extra, w)
    x_min_new = max(x_min - extra, 0)
    y_max_new = min(y_max + extra, h)
    y_min_new = max(y_min - extra, 0)

    img_crop = cv2_img[y_min_new:y_max_new, x_min_new:x_max_new]

    img_crop_proc = skimage.transform.resize(img_crop, (48, 48, 3))
    img_crop_proc = np.expand_dims(img_crop_proc, axis=0)

    prediction = model.predict(img_crop_proc)
    predicted_class_index = np.argmax(prediction)
    letter_predicted = label_dict[predicted_class_index]

    ss.word.append(letter_predicted)

with st.form("word_form"):
    st.markdown("".join(ss.word))
    if st.form_submit_button("Clear Word"):
        ss.word = []  # Clear the word when the button is clicked
