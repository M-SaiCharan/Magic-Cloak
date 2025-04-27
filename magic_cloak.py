# MAGIC CLOAK v1.0

import cv2
import numpy as np
import time

# ------------------ SETTINGS ------------------ #
OUTPUT_FILE = 'mc_output.avi'
BACKGROUND_CAPTURE_DELAY = 3  # seconds to wait before capturing background
SAVE_VIDEO = True
WINDOW_NAME = 'Magic Cloak v1.0'
# ------------------------------------------------ #

def nothing(x):
    pass

# Open webcam
cap = cv2.VideoCapture(0)

# Video writer if saving
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Create a window
cv2.namedWindow(WINDOW_NAME)

# Create trackbars for dynamic HSV range adjustment
cv2.createTrackbar('LH', WINDOW_NAME, 0, 180, nothing)
cv2.createTrackbar('LS', WINDOW_NAME, 120, 255, nothing)
cv2.createTrackbar('LV', WINDOW_NAME, 70, 255, nothing)
cv2.createTrackbar('UH', WINDOW_NAME, 10, 180, nothing)
cv2.createTrackbar('US', WINDOW_NAME, 255, 255, nothing)
cv2.createTrackbar('UV', WINDOW_NAME, 255, 255, nothing)

print("[INFO] Please stay out of frame. Capturing background in 3 seconds...")
time.sleep(BACKGROUND_CAPTURE_DELAY)

# Capture the static background
ret, background = cap.read()
background = np.flip(background, axis=1)
print("[INFO] Background Captured Successfully!")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of trackbars
    l_h = cv2.getTrackbarPos('LH', WINDOW_NAME)
    l_s = cv2.getTrackbarPos('LS', WINDOW_NAME)
    l_v = cv2.getTrackbarPos('LV', WINDOW_NAME)
    u_h = cv2.getTrackbarPos('UH', WINDOW_NAME)
    u_s = cv2.getTrackbarPos('US', WINDOW_NAME)
    u_v = cv2.getTrackbarPos('UV', WINDOW_NAME)

    lower_color = np.array([l_h, l_s, l_v])
    upper_color = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Morphological Transformations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    # Inverse mask
    inverse_mask = cv2.bitwise_not(mask)

    # Segmentation
    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Final output
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow(WINDOW_NAME, final_output)

    # Initialize video writer after frame size known
    if SAVE_VIDEO and out is None:
        height, width = final_output.shape[:2]
        out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 20.0, (width, height))

    if SAVE_VIDEO:
        out.write(final_output)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
if SAVE_VIDEO and out:
    out.release()
cv2.destroyAllWindows()
