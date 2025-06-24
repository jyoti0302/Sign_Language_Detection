import cv2
import os
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Dataset paths
RAW_DATA_PATH = 'raw_dataset'  # Folder with original images
PROCESSED_DATA_PATH = 'cropped_dataset'  # Folder to save cropped hand images

# Create processed data folder
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# Loop through all gesture folders
for gesture_label in os.listdir(RAW_DATA_PATH):
    gesture_path = os.path.join(RAW_DATA_PATH, gesture_label)
    save_path = os.path.join(PROCESSED_DATA_PATH, gesture_label)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image_name in os.listdir(gesture_path):
        image_path = os.path.join(gesture_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]

                h, w, _ = image.shape
                xmin, xmax = int(min(x_list) * w) - 20, int(max(x_list) * w) + 20
                ymin, ymax = int(min(y_list) * h) - 20, int(max(y_list) * h) + 20

                # Clamp to image boundaries
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)

                hand_crop = image[ymin:ymax, xmin:xmax]

                if hand_crop.size == 0:
                    continue

                hand_crop = cv2.resize(hand_crop, (224, 224))
                save_image_path = os.path.join(save_path, image_name)
                cv2.imwrite(save_image_path, hand_crop)
                print(f"[INFO] Saved cropped hand to {save_image_path}")

        else:
            print(f"[WARNING] No hand detected in {image_path}")

print("[INFO] Hand detection and cropping completed for all images.")
