import cv2     #used for video capturing
import mediapipe as mp   #tracks the hand movement
import screen_brightness_control as sbc   #access the systems brightness setting
import numpy as np   # used for calculating the distance between your fingers

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
         
            # Calculate distance between thumb tip and index finger tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate Euclidean distance
            distance = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)

            # Map distance to brightness range (0-100)
            brightness = int(np.interp(distance, [0, 0.2], [0, 100]))  # Adjust 0.2 based on actual distance range
            sbc.set_brightness(brightness)

            # Draw landmarks
            for lm in hand_landmarks.landmark:
                cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])), 5, (0, 255, 0), -1)

    cv2.imshow('Hand Detection for Brightness Control', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
