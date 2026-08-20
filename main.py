import cv2
import mediapipe as mp
import time

import numpy as np

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

landmarker = HandLandmarker.create_from_options(options)
frame_count = 0

def main():
    global frame_count
    print("Starting the hand tracking application...")
    while True:
        attempt = 0
        success, img = cap.read()
        while not success and attempt < 5:
            time.sleep(0.2)
            success, img = cap.read()
            attempt += 1
        if not success:
            print("Failed to capture image from camera after multiple attempts.")
            break

        frame_count += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        result = landmarker.detect_for_video(mp_image, frame_count)

        if result.hand_landmarks:
            fingertip_indices = [4, 8, 12]
            hand_tip_points = []

            for hand_landmarks in result.hand_landmarks:
                tip_points = []
                for idx in fingertip_indices:
                    landmark = hand_landmarks[idx]
                    x = int(landmark.x * img.shape[1])
                    y = int(landmark.y * img.shape[0])
                    tip_points.append((x, y))
                    # cv2.circle(img, (x, y), 8, (255, 0, 0), -1)
                hand_tip_points.append(tip_points)

                

            if len(hand_tip_points) >= 2:
                effects = ["comic_dots", "green_ascii"]
                for finger_index in range(len(fingertip_indices) - 2, -1, -1):
                    p1 = hand_tip_points[0][finger_index]
                    p2 = hand_tip_points[1][finger_index]
                    p3 = hand_tip_points[0][finger_index + 1]
                    p4 = hand_tip_points[1][finger_index + 1]
                    polygon = np.array([[p1, p3, p4, p2]], dtype=np.int32)
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, polygon, 255)

                    effect = effects[finger_index % len(effects)]
                    if effect == "comic_dots":
                        # Comic-book halftone dots with bold color edges.
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        posterized = (img // 64) * 64
                        edges = cv2.Canny(gray, 80, 160)
                        transformed = posterized.copy()
                        transformed[edges > 0] = (0, 0, 0)
                        spacing = 8
                        for y in range(spacing // 2, img.shape[0], spacing):
                            for x in range(spacing // 2, img.shape[1], spacing):
                                radius = max(1, int((255 - int(gray[y, x])) / 100))
                                cv2.circle(transformed, (x, y), radius, (40, 40, 40), -1)
                    else:
                        # Green terminal-style ASCII shading.
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        chars = " .:-=+*#%@"
                        cell_w, cell_h = 8, 12
                        small = cv2.resize(
                            gray,
                            (max(1, img.shape[1] // cell_w), max(1, img.shape[0] // cell_h)),
                        )
                        transformed = np.zeros_like(img)
                        for row in range(small.shape[0]):
                            for col in range(small.shape[1]):
                                char = chars[int(small[row, col]) * len(chars) // 256]
                                cv2.putText(
                                    transformed, char,
                                    (col * cell_w, (row + 1) * cell_h),
                                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 220, 0), 1,
                                )

                    img[mask == 255] = transformed[mask == 255]
                    cv2.polylines(img, [polygon], True,  2)
        else:
            pass
        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()