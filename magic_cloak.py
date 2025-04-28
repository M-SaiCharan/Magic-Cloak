# MAGIC CLOAK v2.0 

import cv2
import numpy as np
import time

# ------------------ SETTINGS ------------------ #
OUTPUT_FILE = 'magic_cloak_output.mp4'
SAVE_VIDEO = True
BACKGROUND_CAPTURE_DELAY = 3
WINDOW_NAME = 'Magic Cloak 2.0'
CLICKED = False
hsv_value = None
MARGIN = 20  # margin for hue range
# ------------------------------------------------ #

def nothing(x):
    pass

def pick_color(event, x, y, flags, param):
    global hsv_value, CLICKED
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_value = hsv[y, x]
        CLICKED = True

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

start_time = time.time()
countdown = BACKGROUND_CAPTURE_DELAY

while countdown > 0:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    frame = np.flip(frame, axis=1)
    
    elapsed = int(time.time() - start_time)
    if elapsed >= 1:
        start_time = time.time()
        countdown -= 1
    
    display_text = f"Capturing Background in {countdown}..." if countdown > 0 else "Capturing Background..."
    frame_display = frame.copy()
    cv2.putText(frame_display, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.imshow(WINDOW_NAME, frame_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()


ret, background = cap.read()
background = np.flip(background, axis=1)
print("[INFO] Background Captured Successfully!")

print("[INFO] Please click on your cloak in the video window...")

while not CLICKED:
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)
    temp_frame = frame.copy()
    cv2.putText(temp_frame, "Click on your cloak!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
    cv2.imshow(WINDOW_NAME, temp_frame)
    cv2.setMouseCallback(WINDOW_NAME, pick_color, frame)
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

print(f"[INFO] Cloak color selected: HSV={hsv_value}")

lower_color = np.array([max(hsv_value[0] - MARGIN, 0), 100, 50])
upper_color = np.array([min(hsv_value[0] + MARGIN, 180), 255, 255])

print(f"[INFO] Lower HSV: {lower_color}, Upper HSV: {upper_color}")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
if SAVE_VIDEO and out:
    out.release()
cv2.destroyAllWindows()
