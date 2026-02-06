#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# ==================== CAU HINH ====================
ESP32_CAM_IP = "192.168.100.12"  # <- SUA IP CUA BAN
ESP32_URL = f"http://{ESP32_CAM_IP}/stream"

# ==================== HAM ARGUMENTS ====================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    args = parser.parse_args()
    return args

# ==================== HAM MAIN ====================
def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    print("\n" + "="*60)
    print("HE THONG NHAN DIEN NGON NGU KY HIEU")
    print("="*60)

    # KHOI TAO CA 2 CAMERA
    current_source = 0  # 0 = ESP32, 1 = PC webcam
    
    # ESP32-CAM
    print(f"Dang ket noi ESP32-CAM: {ESP32_URL}")
    cap_esp32 = cv.VideoCapture(ESP32_URL)
    esp32_available = cap_esp32.isOpened()
    
    if esp32_available:
        print("[OK] ESP32-CAM ket noi thanh cong")
        ret, test_frame = cap_esp32.read()
        if ret:
            h, w = test_frame.shape[:2]
            print(f"   Kich thuoc: {w}x{h}")
    else:
        print("[X] ESP32-CAM khong ket noi duoc")
        print(f"   Kiem tra IP: {ESP32_CAM_IP}")
    
    # PC Webcam
    print(f"\nDang mo PC Webcam...")
    cap_pc = cv.VideoCapture(cap_device)
    cap_pc.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap_pc.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    pc_available = cap_pc.isOpened()
    
    if pc_available:
        print("[OK] PC Webcam mo thanh cong")
    else:
        print("[X] PC Webcam khong mo duoc")
    
    # Kiem tra it nhat 1 nguon co san
    if not esp32_available and not pc_available:
        print("\n[X] Khong co nguon video nao hoat dong!")
        return
    
    # Chon nguon mac dinh
    if esp32_available:
        current_source = 0
        print("\n[!] Mac dinh: ESP32-CAM")
    else:
        current_source = 1
        print("\n[!] Mac dinh: PC Webcam")
    
    print("="*60)

    # Kiem tra model files
    print("\nKiem tra model files...")
    try:
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        print(f"   [OK] Keypoint labels: {len(keypoint_classifier_labels)} chu cai")
        
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            point_history_classifier_labels = [row[0] for row in csv.reader(f)]
        print(f"   [OK] Point history labels: {len(point_history_classifier_labels)} cu chi")
    except FileNotFoundError as e:
        print(f"   [X] LOI: Khong tim thay file: {e}")
        return

    # Khoi tao MediaPipe
    print("\nKhoi tao MediaPipe...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    print("   [OK] MediaPipe ready")

    # Load models
    print("\nLoad AI models...")
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    print("   [OK] Models loaded")

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    
    mode = 0
    number = -1

    print("\nPHIM TAT:")
    print("   ESC : Thoat")
    print("   'c' : CHUYEN DOI NGUON (ESP32 <-> PC Webcam)")
    print("   '0' : Che do nhan dien")
    print("   '1' : Logging keypoint")
    print("   '2' : Logging point history")
    print("   A-Z : Chon label")
    
    print("\n" + "="*60)
    print("HE THONG SAN SANG!")
    print("="*60 + "\n")

    frame_count = 0

    # ==================== VONG LAP CHINH ====================
    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\nDang thoat...")
            break
        
        # CHUYEN DOI NGUON VIDEO
        if key == ord('c') or key == ord('C'):
            # Chi chuyen neu nguon khac co san
            if current_source == 0 and pc_available:
                current_source = 1
                print("\n>>> CHUYEN SANG: PC WEBCAM")
            elif current_source == 1 and esp32_available:
                current_source = 0
                print("\n>>> CHUYEN SANG: ESP32-CAM")
            else:
                print("\n[!] Khong the chuyen doi - nguon khac khong kha dung")
        
        number, mode = select_mode(key, mode)

        # DOC FRAME TU NGUON HIEN TAI
        if current_source == 0:  # ESP32-CAM
            ret, image = cap_esp32.read()
            if not ret:
                print("\n[X] Mat ket noi ESP32-CAM!")
                if pc_available:
                    current_source = 1
                    print("[!] Tu dong chuyen sang PC Webcam")
                else:
                    break
                continue
            use_esp32 = True
        else:  # PC Webcam
            ret, image = cap_pc.read()
            if not ret:
                print("\n[X] Mat ket noi PC Webcam!")
                if esp32_available:
                    current_source = 0
                    print("[!] Tu dong chuyen sang ESP32-CAM")
                else:
                    break
                continue
            # Lat anh khi dung PC webcam
            image = cv.flip(image, 1)
            use_esp32 = False
        
        frame_count += 1
        h, w = image.shape[:2]
        
        # RESIZE ANH NEU QUA NHO
        target_width = 960
        if w < target_width:
            scale = target_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
            w, h = new_w, new_h
        
        debug_image = copy.deepcopy(image)
        
        # TANG CUONG ANH (chi khi dung ESP32)
        if use_esp32:
            alpha = 1.2  # Contrast
            beta = 10    # Brightness
            image_enhanced = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        else:
            image_enhanced = image
        
        # Xu ly MediaPipe
        image_rgb = cv.cvtColor(image_enhanced, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        
        current_letter = ""
        hand_detected = False

        if results.multi_hand_landmarks is not None:
            hand_detected = True
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                    current_letter = keypoint_classifier_labels[hand_sign_id]
                
                if hand_sign_id == 2 and landmark_list and len(landmark_list) > 8:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])
                
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()
                
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                
                finger_gesture_text = ""
                if most_common_fg_id and 0 <= most_common_fg_id[0][0] < len(point_history_classifier_labels):
                    finger_gesture_text = point_history_classifier_labels[most_common_fg_id[0][0]]
                
                debug_image = draw_info_text(debug_image, brect, handedness, current_letter, finger_gesture_text)
        else:
            point_history.append([0, 0])
        
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        
        # HIEN THI NGUON VIDEO (TO, RO RANG)
        source_text = "ESP32-CAM" if use_esp32 else "PC WEBCAM"
        source_color = (0, 255, 255) if use_esp32 else (255, 0, 255)
        
        cv.putText(debug_image, f"SOURCE: {source_text} ({w}x{h})", (10, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(debug_image, f"SOURCE: {source_text} ({w}x{h})", (10, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, source_color, 2, cv.LINE_AA)
        
        # Hien thi huong dan chuyen doi
        cv.putText(debug_image, "Nhan 'C' de chuyen doi nguon", (10, 160),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)
        
        # Hien thi ket qua nhan dien
        if mode == 0:
            if hand_detected:
                if current_letter != "":
                    cv.putText(debug_image, f">>> {current_letter} <<<", (10, 220), 
                              cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 10, cv.LINE_AA)
                    cv.putText(debug_image, f">>> {current_letter} <<<", (10, 220), 
                              cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5, cv.LINE_AA)
                else:
                    cv.putText(debug_image, "Dang phan tich...", (10, 220), 
                              cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv.LINE_AA)
            else:
                cv.putText(debug_image, "Khong thay ban tay", (10, 220), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv.LINE_AA)
        
        cv.imshow('Hand Gesture Recognition - Dual Source', debug_image)

    # Don dep
    if esp32_available:
        cap_esp32.release()
    if pc_available:
        cap_pc.release()
    cv.destroyAllWindows()
    print("\n[OK] Thoat thanh cong!")
    print(f"Thong ke: Da xu ly {frame_count} frames\n")

# ==================== CAC HAM PHU TRO ====================
def select_mode(key, mode):
    number = -1
    if 97 <= key <= 122:
        number = key - 97
    elif 65 <= key <= 90:
        number = key - 65
    
    if key == ord('0'):
        mode = 0
        print("Che do: Nhan dien")
    elif key == ord('1'):
        mode = 1
        print("Che do: Logging Key Point")
    elif key == ord('2'):
        mode = 2
        print("Che do: Logging Point History")
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [np.array((landmark_x, landmark_y))], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    if not landmark_list:
        return []
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    
    for index in range(len(temp_landmark_list)):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    if not temp_landmark_list:
        return []

    max_value = max(map(abs, temp_landmark_list))
    if max_value == 0:
        max_value = 1

    return list(map(lambda n: n / max_value, temp_landmark_list))

def pre_process_point_history(image, point_history):
    if not point_history or not any(ph[0] != 0 or ph[1] != 0 for ph in point_history):
        return []
        
    image_width, image_height = image.shape[1], image.shape[0]
    if image_width == 0 or image_height == 0:
        return []

    temp_point_history = copy.deepcopy(list(point_history))
    
    base_x, base_y = 0, 0
    first_valid_point_found = False
    for point in temp_point_history:
        if point[0] != 0 or point[1] != 0:
            base_x, base_y = point[0], point[1]
            first_valid_point_found = True
            break
    
    if not first_valid_point_found:
        return list(itertools.chain.from_iterable(temp_point_history))

    processed_history = []
    for point in temp_point_history:
        norm_x = (point[0] - base_x) / image_width
        norm_y = (point[1] - base_y) / image_height
        processed_history.append([norm_x, norm_y])
        
    return list(itertools.chain.from_iterable(processed_history))

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode not in [1, 2]:
        return

    if not (0 <= number <= 25):
        return
    
    csv_path_map = {
        1: ('model/keypoint_classifier/keypoint.csv', landmark_list),
        2: ('model/point_history_classifier/point_history.csv', point_history_list)
    }
    csv_path, data_list = csv_path_map[mode]
    
    if not data_list:
        return
    try:
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *data_list])
    except Exception as e:
        print(f"Loi khi ghi vao {csv_path}: {e}")
    return

def draw_landmarks(image, landmark_point):
    if not landmark_point or len(landmark_point) < 21:
        return image

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    for start_idx, end_idx in connections:
        if start_idx < len(landmark_point) and end_idx < len(landmark_point):
            start_point = tuple(map(int, landmark_point[start_idx]))
            end_point = tuple(map(int, landmark_point[end_idx]))
            cv.line(image, start_point, end_point, (0, 0, 0), 6)
            cv.line(image, start_point, end_point, (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        point = tuple(map(int, landmark))
        radius = 8 if index in [4, 8, 12, 16, 20] else 5
        
        cv.circle(image, point, radius, (255, 255, 255), -1)
        cv.circle(image, point, radius, (0, 0, 0), 1)

    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect and brect and len(brect) == 4:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    if not brect or len(brect) != 4:
        return image
    
    hand_label = ""
    if handedness and handedness.classification and handedness.classification[0]:
        hand_label = handedness.classification[0].label

    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = hand_label
    if hand_sign_text != "":
        info_text = f"{info_text}:{hand_sign_text}"
    
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, f"Finger Gesture:{finger_gesture_text}", (10, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, f"Finger Gesture:{finger_gesture_text}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 or point[1] != 0:
            cv.circle(image, tuple(map(int, point)), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, f"FPS:{int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS:{int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 2, cv.LINE_AA)

    mode_strings = {
        0: "Detect",
        1: "Log KeyPoint",
        2: "Log PointHistory"
    }
    current_mode_string = mode_strings.get(mode, "Unknown")

    cv.putText(image, f"MODE: {current_mode_string}", (10, 90),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    if mode in [1, 2] and (0 <= number <= 25):
        label_char = chr(ord('A') + number)
        cv.putText(image, f"LABEL: {label_char} ({number})", (10, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    return image


if __name__ == '__main__':
    main()
