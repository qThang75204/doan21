#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time
import requests

import cv2 as cv
import numpy as np
import mediapipe as mp
import serial
# import serial.tools.list_ports # Bạn có thể giữ lại nếu muốn tự động tìm cổng

# Giả sử utils.py và model đã có sẵn và đúng đường dẫn
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


import threading
use_esp32 = False


esp_frame = None
esp_running = True

def esp32_capture():
    global esp_frame, esp_running

    cap_esp = cv.VideoCapture(ESP32_URL)

    while esp_running:
        ret, frame = cap_esp.read()
        if ret:
            esp_frame = frame.copy()

    cap_esp.release()

ESP32_URL = "http://192.168.100.12/stream"   # IP ESP32 của bạn
pc_cam_index = 0                            # webcam laptop
current_cam = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float, # Nên là float
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float, # Nên là float cho mediapipe
                        default=0.5)
    args = parser.parse_args()
    return args


# ------------------ PHẦN MỚI: GỬI QUA HTTP ĐẾN ESP32-CAM ------------------
ESP32_CAM_IP = "http://192.168.100.12"  # ← SỬA THÀNH IP THỰC TẾ CỦA ESP32-CAM (xem Serial Monitor)

last_sent_http_time = 00
http_send_interval = 1.0  # Gửi tối đa 1 lần/giây

def send_letter_to_esp32_cam(letter_index):
    global last_sent_http_time

    current_time = time.time()
    if current_time - last_sent_http_time < 0.5:
        return

    try:
        url = f"http://{ESP32_CAM_IP}/set_letter"
        payload = str(letter_index)

        response = requests.post(
            url,
            data=payload,
            timeout=1,
            headers={"Connection": "close"}
        )

        if response.status_code == 200:
            print("Sent:", payload)
            last_sent_http_time = current_time

    except Exception as e:
        print("ESP32 timeout:", e)
        
def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    current_cam = 0   # 0 = PC, 1 = ESP32

    cap = cv.VideoCapture(pc_cam_index)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    t = threading.Thread(target=esp32_capture, daemon=True)
    t.start()


    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    try:
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            point_history_classifier_labels = [row[0] for row in csv.reader(f)]
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file CSV label: {e}. Vui lòng kiểm tra đường dẫn.")
        return

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0
    number = -1
    previous_letter = ""
    letter_stable_count = 0
    min_stable_frames = 10
    last_sent_time = 0
    send_interval = 1.0 
    
    binary_representation_5bit = "" # Khởi tạo biến để lưu trữ chuỗi nhị phân

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('c'):
             current_cam = 1 - current_cam
             if current_cam == 0:
                 print(">> PC CAMERA")
             else:
                 print(">> ESP32 CAMERA")

        if current_cam == 0:
             ret, image = cap.read()
             if not ret:
                 continue
        else:
             if esp_frame is None:
                 continue
             image = esp_frame.copy()
        if key == 27: break # Phím ESC để thoát
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret: break
        image = cv.flip(image, 1) # Lật ảnh để giống như gương
        debug_image = copy.deepcopy(image)

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Mediapipe cần ảnh RGB
        image_rgb.flags.writeable = False # Tối ưu hóa: đánh dấu ảnh không thể ghi đè để tăng tốc xử lý
        results = hands.process(image_rgb)
        # image_rgb.flags.writeable = True # Không cần thiết nếu không sửa image_rgb nữa

        current_letter = ""
        letter_index_for_display = 0 # Để lưu letter_index cho hiển thị

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                    current_letter = keypoint_classifier_labels[hand_sign_id]
                else:
                    current_letter = "" # Hoặc "" nếu ID không hợp lệ
                
                # Giả sử ID 2 là "Point gesture", kiểm tra landmark_list[8] tồn tại
                if hand_sign_id == 2 and landmark_list and len(landmark_list) > 8: 
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0,0]) # Nếu không phải point gesture hoặc không có landmark_list[8]

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (history_length * 2): # *2 vì mỗi điểm có x,y
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common(1) # Lấy 1 phần tử phổ biến nhất
                
                fg_label = ""
                if most_common_fg_id and 0 <= most_common_fg_id[0][0] < len(point_history_classifier_labels):
                    fg_label = point_history_classifier_labels[most_common_fg_id[0][0]]

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, current_letter, fg_label)
        else:
            point_history.append([0, 0])
            binary_representation_5bit = "" # Reset nếu không có tay

        current_time = time.time()
        if current_letter and current_letter.isalpha() and len(current_letter) == 1:
            letter_index_for_display = ord(current_letter.upper()) - ord('A') + 1
            # Chuyển đổi letter_index thành chuỗi nhị phân 5 bit
            if 1 <= letter_index_for_display <= 31: # 2^5 = 32, đủ cho 1-26
                 binary_representation_5bit = format(letter_index_for_display, '05b')
            else:
                 binary_representation_5bit = "N/A"


            if current_letter.upper() == previous_letter:
                letter_stable_count += 1
            else:
                previous_letter = current_letter.upper()
                letter_stable_count = 1
            
            if letter_stable_count >= min_stable_frames and (current_time - last_sent_time) >= send_interval:
               send_letter_to_esp32_cam(letter_index_for_display)
               letter_stable_count = 0 
               last_sent_time = current_time
        else:
            previous_letter = ""
            letter_stable_count = 0
            binary_representation_5bit = "" # Reset nếu chữ không hợp lệ

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        
        # Hiển thị thông tin chữ cái và dạng nhị phân 5-bit
        if current_letter: # Chỉ hiển thị nếu có current_letter
            cv.putText(debug_image, f"Letter: {current_letter.upper()} ({letter_index_for_display if letter_index_for_display > 0 else 'N/A'})", 
                       (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
            if binary_representation_5bit: # Chỉ hiển thị nếu có chuỗi binary
                cv.putText(debug_image, f"Binary (5-bit): {binary_representation_5bit}", (10, 160), # Tọa độ y=160
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow('Hand Gesture Recognition', debug_image)
        

   

    
        
    cap.release()
    cv.destroyAllWindows()

def select_mode(key, mode):
    number = -1 # Mặc định number là -1 (không có số nào được chọn cho logging)
    if 97 <= key <= 122: # chữ thường từ 'a' tới 'z'
        number = key - 97  # a = 0, b = 1, ..., z = 25
    elif 65 <= key <= 90: # chữ hoa từ 'A' tới 'Z'
        number = key - 65  # A = 0, B = 1, ..., Z = 25
    
    if key == ord('0'): 
        mode = 0
        print("Chế độ: Nhận diện & Gửi (Không Logging)")
    elif key == ord('1'): 
        mode = 1
        print("Chế độ: Logging Key Point (Nhấn phím chữ A-Z để chọn label)")
    elif key == ord('2'): 
        mode = 2
        print("Chế độ: Logging Point History (Nhấn phím chữ A-Z để chọn label)")
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark): # Sử dụng _ nếu không dùng index
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [np.array((landmark_x, landmark_y))], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark): # Sử dụng _ nếu không dùng index
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    if not landmark_list: return [] # Trả về rỗng nếu list rỗng
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Giả sử landmark_list[0] luôn tồn tại nếu landmark_list không rỗng
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    
    for index in range(len(temp_landmark_list)): # Lặp qua index
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    if not temp_landmark_list: return [] # Trả về rỗng nếu list rỗng sau chain

    max_value = max(map(abs, temp_landmark_list)) # Dùng map(abs, ...)
    if max_value == 0: max_value = 1 # Tránh chia cho 0

    return list(map(lambda n: n / max_value, temp_landmark_list))

def pre_process_point_history(image, point_history):
    # Kiểm tra point_history có dữ liệu thực sự (khác (0,0))
    if not point_history or not any(ph[0] != 0 or ph[1] != 0 for ph in point_history): 
        return [] 
        
    image_width, image_height = image.shape[1], image.shape[0]
    if image_width == 0 or image_height == 0: return [] # Tránh chia cho 0

    temp_point_history = copy.deepcopy(list(point_history)) # Chuyển deque thành list để deepcopy
    
    base_x, base_y = 0, 0
    first_valid_point_found = False
    for point in temp_point_history:
        if point[0] != 0 or point[1] != 0: # Tìm điểm đầu tiên khác (0,0)
            base_x, base_y = point[0], point[1]
            first_valid_point_found = True
            break
    
    if not first_valid_point_found: # Nếu tất cả các điểm là (0,0)
        # Trả về list các số 0 đã được flatten, hoặc list rỗng tùy theo logic mong muốn
        # Hiện tại, nó sẽ trả về list flatten của các cặp [0,0]
        return list(itertools.chain.from_iterable(temp_point_history)) 

    processed_history = []
    for point in temp_point_history:
        norm_x = (point[0] - base_x) / image_width
        norm_y = (point[1] - base_y) / image_height
        processed_history.append([norm_x, norm_y])
        
    return list(itertools.chain.from_iterable(processed_history))

def logging_csv(number, mode, landmark_list, point_history_list):
    # Với mode = 0 (hoặc mode không hợp lệ), không thực hiện ghi dữ liệu
    if mode not in [1, 2]:
        return

    # Chỉ ghi nếu number là một label hợp lệ (0-25 tương ứng A-Z)
    if not (0 <= number <= 25):
        return
    
    csv_path_map = {
        1: ('model/keypoint_classifier/keypoint.csv', landmark_list),
        2: ('model/point_history_classifier/point_history.csv', point_history_list)
    }
    csv_path, data_list = csv_path_map[mode] 
    
    if not data_list: # Không ghi nếu danh sách dữ liệu rỗng
        return
    try:
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *data_list])
    except Exception as e:
        print(f"Lỗi khi ghi vào {csv_path}: {e}")
    return

def draw_landmarks(image, landmark_point):
    # Kiểm tra landmark_point không rỗng và có đủ 21 điểm
    if not landmark_point or len(landmark_point) < 21:
        return image # Trả về ảnh gốc nếu không có đủ landmark

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Ngón cái
        (0, 5), (5, 6), (6, 7), (7, 8),  # Ngón trỏ
        (5, 9), (9, 10), (10, 11), (11, 12), # Ngón giữa (nối từ 5 hoặc 0)
        (9, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út (nối từ 9 hoặc 0)
        (13, 17), (17, 18), (18, 19), (19, 20), # Ngón út (nối từ 13 hoặc 0)
        (0, 17) # Đường nối lòng bàn tay (wrist to pinky base)
    ] # Các cặp index để vẽ đường nối

    for start_idx, end_idx in connections:
        # Đảm bảo các index nằm trong phạm vi của landmark_point
        if start_idx < len(landmark_point) and end_idx < len(landmark_point):
            start_point = tuple(map(int, landmark_point[start_idx]))
            end_point = tuple(map(int, landmark_point[end_idx]))
            cv.line(image, start_point, end_point, (0, 0, 0), 6) # Viền đen
            cv.line(image, start_point, end_point, (255, 255, 255), 2) # Đường trắng

    # Vẽ các điểm landmark
    for index, landmark in enumerate(landmark_point):
        point = tuple(map(int, landmark))
        radius = 8 if index in [4, 8, 12, 16, 20] else 5 # Đầu ngón tay to hơn
        
        cv.circle(image, point, radius, (255, 255, 255), -1) # Chấm trắng
        cv.circle(image, point, radius, (0, 0, 0), 1) # Viền đen cho chấm

    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect and brect and len(brect) == 4: # Kiểm tra brect hợp lệ
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    if not brect or len(brect) != 4: return image # Kiểm tra brect
    
    hand_label = ""
    if handedness and handedness.classification and handedness.classification[0]:
        hand_label = handedness.classification[0].label

    # Vẽ hình chữ nhật nền cho text
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = hand_label
    if hand_sign_text != "": # Chỉ thêm dấu : nếu có hand_sign_text
        info_text = f"{info_text}:{hand_sign_text}"
    
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "": # Chỉ vẽ nếu có finger_gesture_text
        cv.putText(image, f"Finger Gesture:{finger_gesture_text}", (10, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA) 
        cv.putText(image, f"Finger Gesture:{finger_gesture_text}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 or point[1] != 0: # Chỉ vẽ nếu điểm không phải là (0,0)
            cv.circle(image, tuple(map(int,point)), 1 + int(index / 2),
                      (152, 251, 152), 2) # Màu xanh lá nhạt
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, f"FPS:{int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
               0.8, (0, 0, 0), 4, cv.LINE_AA) 
    cv.putText(image, f"FPS:{int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 2, cv.LINE_AA)

    mode_strings = {
        0: "Detect & Send",
        1: "Log KeyPoint",
        2: "Log PointHistory"
    }
    current_mode_string = mode_strings.get(mode, "Unknown Mode") # Lấy chuỗi mode, mặc định là "Unknown"

    cv.putText(image, f"MODE: {current_mode_string}", (10, 90), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    if mode in [1, 2] and (0 <= number <= 25): # Nếu đang ở mode logging và có label hợp lệ (A-Z)
        label_char = chr(ord('A') + number)
        cv.putText(image, f"LABEL: {label_char} ({number})", (10, 110), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()