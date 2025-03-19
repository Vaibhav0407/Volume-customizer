import cv2
import mediapipe as mp
import math
import pyautogui

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Open the camera
cap = cv2.VideoCapture(0)

# Get screen width and height
screen_width, screen_height = pyautogui.size()

# Initialize previous hand position5
prev_hand_pos = (0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to find hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks for the hand
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

            # Calculate the relative position of the hand
            rel_x = cx / w
            rel_y = cy / h

            # Calculate the volume based on the relative position of the hand
            volume = rel_y

            # Update the system volume
            pyautogui.press('volumedown') if rel_x < prev_hand_pos[0] else pyautogui.press('volumeup')
            prev_hand_pos = (rel_x, rel_y)

    # Show the frame
    cv2.imshow("Hand Tracking", frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()