#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp

# Giả sử bạn có các lớp này trong thư mục 'model'
# Nếu chúng ở file khác, hãy đảm bảo import đúng cách
try:
    from model import KeyPointClassifier
except ImportError:
    print("Cảnh báo: Không tìm thấy module 'model'. Đảm bảo KeyPointClassifier có sẵn.")
    # Định nghĩa lớp giả để code chạy không lỗi nếu thiếu file
    class KeyPointClassifier:
        def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite'):
            print(f"Khởi tạo KeyPointClassifier giả lập (cần file thực tế: {model_path})")
            # Bạn cần thay thế bằng việc load model thực tế ở đây
            # Ví dụ: self.interpreter = tf.lite.Interpreter(model_path=model_path)
            # self.interpreter.allocate_tensors()
            # self.input_details = self.interpreter.get_input_details()
            # self.output_details = self.interpreter.get_output_details()
            pass
        def __call__(self, landmark_list):
            # Hàm giả lập, luôn trả về ID 0
            # Thay thế bằng logic inference thực tế
            # Ví dụ:
            # input_details_tensor_index = self.input_details[0]['index']
            # self.interpreter.set_tensor(
            #     input_details_tensor_index,
            #     np.array([landmark_list], dtype=np.float32))
            # self.interpreter.invoke()
            # output_details_tensor_index = self.output_details[0]['index']
            # result = self.interpreter.get_tensor(output_details_tensor_index)
            # result_index = np.argmax(np.squeeze(result))
            # return result_index
            print("Đang sử dụng KeyPointClassifier giả lập, trả về ID 0")
            return 0

# --- Các hàm xử lý phụ trợ (giữ lại những hàm cần thiết) ---

def calc_landmark_list(image, landmarks):
    """Tính toán danh sách tọa độ điểm mốc trên ảnh gốc."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    """Tiền xử lý danh sách điểm mốc (chuẩn hóa)."""
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0: # Sử dụng điểm gốc cổ tay (wrist) làm gốc tọa độ
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Chuyển đổi thành danh sách phẳng
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Chuẩn hóa dựa trên giá trị tuyệt đối lớn nhất
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        if max_value == 0: # Tránh chia cho 0
            return 0
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def draw_landmarks(image, landmark_point):
    """Vẽ các điểm mốc và đường nối lên ảnh."""
    if len(landmark_point) > 0:
        # Vẽ các đường nối trên lòng bàn tay và các ngón
        # (Mã vẽ chi tiết giống như trong code gốc của bạn)
        # Thumb
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)
        # Index finger
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[5]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)
        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)
        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)
        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)
        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[5]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # Vẽ các điểm landmark
    for index, landmark in enumerate(landmark_point):
        # Vẽ các khớp ngón tay và cổ tay
        if index in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # Vẽ các đầu ngón tay
        if index in [4, 8, 12, 16, 20]:
             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def draw_info_text(image, handedness, hand_sign_text):
    """Vẽ thông tin tay (trái/phải) và tên cử chỉ lên ảnh."""
    # Lấy thông tin tay (Left/Right)
    hand_label = handedness.classification[0].label

    # Chuẩn bị text hiển thị
    info_text = f"{hand_label}: {hand_sign_text}"

    # Vẽ chữ lên góc trên bên trái
    cv.putText(image, info_text, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA) # Viền đen
    cv.putText(image, info_text, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA) # Chữ trắng

    return image

# --- Hàm Main ---
def main():
    # --- Cấu hình Cố định ---
    cap_device = 0 # ID camera mặc định
    cap_width = 960
    cap_height = 540
    use_static_image_mode = False # Chế độ video stream
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5
    max_num_hands = 1 # Chỉ nhận diện 1 bàn tay

    # --- Khởi tạo Camera ---
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # --- Khởi tạo MediaPipe Hands ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # --- Khởi tạo Bộ phân loại Cử chỉ Tĩnh ---
    try:
        keypoint_classifier = KeyPointClassifier(model_path='model/keypoint_classifier/keypoint_classifier.tflite')
    except Exception as e:
        print(f"Lỗi khi khởi tạo KeyPointClassifier: {e}")
        print("Kiểm tra lại đường dẫn và file model.")
        return # Thoát nếu không load được model

    # --- Đọc Nhãn Cử chỉ Tĩnh ---
    keypoint_labels_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    try:
        with open(keypoint_labels_path, encoding='utf-8-sig') as f:
            # Đọc file CSV và lấy cột đầu tiên làm nhãn
            import csv # Import ở đây vì chỉ dùng 1 lần
            keypoint_classifier_labels = [
                row[0] for row in csv.reader(f) if row # Bỏ qua dòng trống nếu có
            ]
            print(f"Đã đọc {len(keypoint_classifier_labels)} nhãn từ: {keypoint_labels_path}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file nhãn: {keypoint_labels_path}")
        # Tạo danh sách nhãn giả lập nếu file không tồn tại
        keypoint_classifier_labels = [f"Gesture_{i}" for i in range(10)] # Giả sử có 10 cử chỉ
        print(f"Đang sử dụng danh sách nhãn giả lập: {keypoint_classifier_labels}")
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file nhãn: {e}")
        return

    # --- Vòng lặp chính ---
    while True:
        # Xử lý phím nhấn (chỉ để thoát)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # --- Đọc khung hình từ Camera ---
        ret, image = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc khung hình từ camera.")
            break
        image = cv.flip(image, 1)  # Lật ảnh (hiệu ứng gương)
        debug_image = copy.deepcopy(image) # Tạo bản sao để vẽ lên

        # --- Phát hiện bàn tay ---
        # Chuyển ảnh sang RGB vì MediaPipe cần RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Đánh dấu không thể ghi để tối ưu
        image_rgb.flags.writeable = False
        # Thực hiện phát hiện
        results = hands.process(image_rgb)
        # Đánh dấu có thể ghi lại
        image_rgb.flags.writeable = True

        # --- Xử lý kết quả nếu phát hiện được tay ---
        if results.multi_hand_landmarks:
            # Lặp qua từng bàn tay được phát hiện (dù chỉ cấu hình cho 1 tay)
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                # --- Tính toán và tiền xử lý điểm mốc ---
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # --- Phân loại cử chỉ tay tĩnh ---
                try:
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    # Lấy tên cử chỉ từ ID
                    if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                         hand_sign_text = keypoint_classifier_labels[hand_sign_id]
                    else:
                         hand_sign_text = "Unknown ID"
                         print(f"Cảnh báo: ID cử chỉ ({hand_sign_id}) nằm ngoài phạm vi nhãn.")

                    # In kết quả ra console (ID và Tên cử chỉ)
                    print(f"Detected Hand: {handedness.classification[0].label}, Gesture ID: {hand_sign_id}, Gesture Name: {hand_sign_text}")

                    # --- Vẽ lên ảnh debug ---
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(debug_image, handedness, hand_sign_text)

                except Exception as e:
                    print(f"Lỗi trong quá trình phân loại hoặc vẽ: {e}")


        # --- Hiển thị ảnh debug ---
        cv.imshow('Hand Gesture Recognition (Simplified)', debug_image)

    # --- Dọn dẹp ---
    cap.release()
    cv.destroyAllWindows()
    print("Chương trình kết thúc.")


if __name__ == '__main__':
    main()