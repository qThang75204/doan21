import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Thêm thư viện để vẽ đồ thị
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

RANDOM_SEED = 42

# --- Các đường dẫn và hằng số không đổi ---
dataset = 'model/keypoint_classifier/keypoint.csv'
# Đổi phần mở rộng file lưu model từ .hdf5 sang .keras (Giữ nguyên theo code gốc)
model_save_path = 'model/keypoint_classifier/keypoint_classifier.keras'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
NUM_CLASSES = 26
INPUT_SHAPE = 21 * 2 # Tính toán kích thước input

# --- Đọc và chuẩn bị dữ liệu (Không đổi) ---
print("Loading dataset...")
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, INPUT_SHAPE + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

# --- Định nghĩa kiến trúc mô hình mới ---
print("Building model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((INPUT_SHAPE, )),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary() # In cấu trúc mô hình

# --- Callbacks (Thêm ReduceLROnPlateau) ---
# Model checkpoint callback (Không đổi)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy', # Lưu model dựa trên val_accuracy tốt nhất
    verbose=1,
    save_best_only=True, # Chỉ lưu model tốt nhất
    save_weights_only=False,
    mode='max' # Chế độ 'max' vì ta muốn tối đa hóa accuracy
)

# Callback for early stopping (Không đổi, có thể tăng patience nếu muốn huấn luyện lâu hơn)
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Theo dõi val_loss
    patience=30,        # Tăng patience một chút
    verbose=1,
    restore_best_weights=True # Khôi phục trọng số tốt nhất khi dừng sớm
)

# **Mới**: Callback for reducing learning rate
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', # Theo dõi val_loss
    factor=0.2,         # Giảm LR đi 5 lần (factor=0.2)
    patience=10,        # Giảm LR nếu val_loss không cải thiện sau 10 epochs
    verbose=1,
    min_lr=1e-6         # Learning rate tối thiểu
)

# --- Model Compilation (Không đổi) ---
print("Compiling model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Model Training (Sử dụng callbacks mới và lưu history) ---
print("Starting training...")
history = model.fit(
    X_train,
    y_train,
    epochs=1000, # Giữ nguyên số epochs tối đa, EarlyStopping sẽ dừng khi cần
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback, lr_scheduler] # Thêm lr_scheduler vào callbacks
)
print("Training finished.")

# --- **Mới**: Hàm vẽ biểu đồ Accuracy và Loss ---
def plot_history(history):
    """Vẽ biểu đồ accuracy và loss từ đối tượng history của Keras."""
    print("Plotting training history...")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5)) # Tạo 1 hàng, 2 cột subplot

    # Vẽ Accuracy
    axs[0].plot(history.history['accuracy'], label='Training Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='lower right')
    axs[0].grid(True) # Thêm lưới cho dễ nhìn

    # Vẽ Loss
    axs[1].plot(history.history['loss'], label='Training Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    axs[1].grid(True) # Thêm lưới

    plt.tight_layout() # Tự động điều chỉnh khoảng cách subplot
    plt.show() # Hiển thị biểu đồ

# --- Gọi hàm vẽ biểu đồ ---
plot_history(history)

# --- Model Evaluation (Sử dụng model tốt nhất đã được khôi phục bởi EarlyStopping hoặc load từ checkpoint) ---
# Lưu ý: Nếu dùng restore_best_weights=True trong EarlyStopping, model hiện tại đã là model tốt nhất
# Nếu không, hoặc muốn chắc chắn, hãy load lại model từ checkpoint
print("Loading the best model for evaluation...")
# model = tf.keras.models.load_model(model_save_path) # Dòng này không cần thiết nếu dùng restore_best_weights=True

print("Evaluating model on test data...")
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
print(f"Test Loss: {val_loss:.4f}")
print(f"Test Accuracy: {val_acc:.4f}")

# --- Inference Test (Không đổi) ---
print("Running inference test on first test sample...")
predict_result = model.predict(np.array([X_test[0]]))
print(f"Raw prediction output: {np.squeeze(predict_result)}")
print(f"Predicted class index: {np.argmax(np.squeeze(predict_result))}")
print(f"Actual class index: {y_test[0]}")

# --- Confusion Matrix and Classification Report (Không đổi) ---
def print_confusion_matrix(y_true, y_pred, report=True):
    """Hàm in confusion matrix và classification report (Không đổi)."""
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(10, 8)) # Tăng kích thước để dễ đọc hơn
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False, cmap='Blues', ax=ax) # Thêm cmap
    ax.set_ylim(len(set(y_true)), 0)
    ax.set_xlabel("Predicted Label") # Thêm nhãn trục
    ax.set_ylabel("True Label")      # Thêm nhãn trục
    ax.set_title("Confusion Matrix") # Thêm tiêu đề
    plt.show()

    if report:
        print('\nClassification Report')
        # Lấy tên nhãn (nếu có) hoặc dùng số trực tiếp
        # target_names = [f'Class {i}' for i in labels] # Ví dụ nếu muốn tên rõ ràng hơn
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0)) # Thêm zero_division

print("Calculating predictions for confusion matrix...")
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)

# --- Save as a model dedicated to inference (Không đổi) ---
# Lưu ý: Model được lưu ở đây sẽ là model cuối cùng sau khi huấn luyện,
# hoặc là model tốt nhất nếu EarlyStopping có restore_best_weights=True.
# Nếu bạn muốn chắc chắn lưu model tốt nhất từ checkpoint, hãy load lại trước khi save:
# model = tf.keras.models.load_model(model_save_path)
print(f"Saving final model for inference (without optimizer) to {model_save_path}...")
model.save(model_save_path, include_optimizer=False)
print("Model saved.")
# === EXPORT SANG SAVEDMODEL (BẮT BUỘC CHO KERAS 3 + TFLITE) ===
saved_model_dir = "model/keypoint_classifier/saved_model"

print(f"Exporting model to SavedModel at {saved_model_dir}...")
model.export(saved_model_dir)
print("SavedModel exported.")
# --- Transform model (quantization) (Không đổi) ---
print(f"Converting model to TFLite (quantized) and saving to {tflite_save_path}...")
# Load lại model vừa lưu để đảm bảo tính nhất quán (đặc biệt nếu không dùng restore_best_weights)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)
print("TFLite model saved.")

# --- TFLite Inference Test (Không đổi) ---
print("Testing TFLite model...")
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

# Get I / O tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Chuẩn bị dữ liệu đầu vào (cần đúng dtype mà TFLite model mong đợi sau quantization)
# Thường là float32 cho input trừ khi có quantization input/output
input_data = np.array([X_test[0]], dtype=np.float32) # Đảm bảo dtype là float32
interpreter.set_tensor(input_details[0]['index'], input_data)

# Inference implementation
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

print(f"TFLite raw prediction output: {np.squeeze(tflite_results)}")
print(f"TFLite predicted class index: {np.argmax(np.squeeze(tflite_results))}")
print(f"Actual class index: {y_test[0]}")

print("\nScript finished successfully!")