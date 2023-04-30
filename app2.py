import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import PIL


# load the model
cnn_model = load_model("./models/facial_landmark_model")

# Get frontal face haar cascade
face_cascade = cv2.CascadeClassifier(
    "C:\\Users\\hp\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
)

# Get webcam
cap = cv2.VideoCapture(0)

# width, it has id numebr 3
cap.set(3, 480)
# height, it has id numebr 4
cap.set(4, 480)

# Glasses filter
sunglasses = cv2.imread("./input/sunglasses.png", -1)

def paste_glass(glass_width, glass_height, img, y, x, brow_coords):
    for i in range(0, glass_width):
        for j in range(0, glass_height):
            if glasses[i, j][3] != 0: 
                img[
                    brow_coords[1] + i + y,
                    left_eye_coords[0] + j + x - int(1.5 * glasses_width),
                ] = glasses[i][j]  
    return img

while True:
    # read data from camera
    success, img = cap.read()

    # preprocess the input by convert BGR values to grayscale, detect faces in the screen
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get coordinates array for the detected faces bounding box
    gray_faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
          
    if len(gray_faces):
        img_copy = np.copy(img) 
        for x, y, width, height in gray_faces:
            gray_face = gray_img[y : y + height, x : x + width]

            # get original width for the detected face
            original_width = gray_face.shape[1]

            # get original height for the detected face
            original_height = gray_face.shape[0]

            # resize gray face image to size 96x96
            resized_gray_face = cv2.resize(gray_face, (96, 96))

            # normalize the image
            resized_gray_face = resized_gray_face / 255

            # reshape image to (1, 96, 96, 1) to pass it to the model (96, 96) => (height, width)
            img_for_prediction = np.reshape(resized_gray_face, (1, 96, 96, 1))
            landmarks = cnn_model.predict(img_for_prediction)[
                0
            ]  # Predict keypoints for the current input

            # slice x axis coordinates
            x_coords = landmarks[0::2]

            # slice y axis coordinates
            y_coords = landmarks[1::2]

            # denormalize the coordinates
            x_coords_denormalized = (x_coords) * original_width
            y_coords_denormalized = (y_coords) * original_height

            # get left and right eyes coordinates to calculate glasses width
            left_eye_coords = (int(x_coords_denormalized[3]), int(y_coords_denormalized[3]))
            right_eye_coords = (
                int(x_coords_denormalized[5]),
                int(y_coords_denormalized[5]),
            )

            # get
            brow_coords = (int(x_coords_denormalized[6]), int(y_coords_denormalized[6]))

            # Scale filter according to keypoint coordinates
            glasses_width = left_eye_coords[0] - right_eye_coords[0]

            img_copy = cv2.cvtColor(
                img_copy, cv2.COLOR_BGR2BGRA
            )  # Used for transparency overlay of filter using the alpha channel

            glasses = sunglasses.copy()
            #'left_eyebrow_outer_end_y'
            glasses_height = int( y_coords_denormalized[5] - y_coords_denormalized[9])
            glasses = cv2.resize(glasses, (glasses_width * 2, glasses_height + 15))
            glass_width, glass_height, gchannels = glasses.shape

            #paste glasse on face 
            img_copy = paste_glass(glass_width, glass_height, img_copy, y, x, brow_coords)  
            
            #revert the image to RGB
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGRA2BGR)  

        cv2.imshow("Output", img_copy)   
    else:
        cv2.imshow("Output", img)  

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 