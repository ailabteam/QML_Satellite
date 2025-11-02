# 01_data_preprocessing.py (v2 - Sửa lỗi inhomogeneous shape)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tqdm
import pickle

# --- CONFIGURATION ---
TRAIN_DIR = 'train'
TEST_DIR = 'test'
PROCESSED_DATA_DIR = 'processed_data'
FIGURES_DIR = 'figures'

WINDOW_SIZE = 5
TRAIN_RATIO_FOR_VALIDATION = 0.1

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("--- Giai đoạn 2: Khám phá và Tiền xử lý Dữ liệu (v2) ---")

# --- BƯỚC 2.1: TẢI, CẮT BỚT VÀ GỘP DỮ LIỆU ---
def load_and_process_data(train_path, test_path):
    train_files = sorted([f for f in os.listdir(train_path) if f.endswith('.npy')])
    test_files = sorted([f for f in os.listdir(test_path) if f.endswith('.npy')])
    
    assert train_files == test_files, "Tên file trong train và test không khớp!"
    channel_names = [f.replace('.npy', '') for f in train_files]
    print(f"Tìm thấy {len(channel_names)} kênh.")

    train_data_list = []
    test_data_list = []
    
    # === THAY ĐỔI 1: Tải tất cả dữ liệu vào list trước ===
    print("Đang tải dữ liệu thô...")
    for filename in tqdm.tqdm(train_files):
        train_d = np.load(os.path.join(train_path, filename))
        test_d = np.load(os.path.join(test_path, filename))
        
        train_data_list.append(train_d[:, 0] if train_d.ndim > 1 else train_d)
        test_data_list.append(test_d[:, 0] if test_d.ndim > 1 else test_d)
        
    # === THAY ĐỔI 2: Tìm độ dài ngắn nhất và cắt bớt ===
    min_train_len = min(len(d) for d in train_data_list)
    min_test_len = min(len(d) for d in test_data_list)
    
    print(f"\nĐộ dài chuỗi train ngắn nhất: {min_train_len}")
    print(f"Độ dài chuỗi test ngắn nhất: {min_test_len}")

    train_truncated = [d[:min_train_len] for d in train_data_list]
    test_truncated = [d[:min_test_len] for d in test_data_list]
    
    # === THAY ĐỔI 3: Gộp lại sau khi đã có cùng độ dài ===
    train_data = np.array(train_truncated).T
    test_data = np.array(test_truncated).T
    
    return train_data, test_data, channel_names

train_data, test_data, CHANNELS = load_and_process_data(TRAIN_DIR, TEST_DIR)

print(f"\nHình dạng dữ liệu train đã gộp: {train_data.shape}")
print(f"Hình dạng dữ liệu test đã gộp: {test_data.shape}")

# Tải nhãn bất thường
anomaly_labels = pd.read_csv('labeled_anomalies.csv')
print("\nThông tin các chuỗi bất thường đã được gán nhãn:")
print(anomaly_labels.head())

# --- Phần còn lại của script không thay đổi ---

# --- BƯỚC 2.2: TRỰC QUAN HÓA (LƯU RA FILE) ---
def plot_channel_data(channel_name, train_data, test_data, anomaly_labels, channels, save_dir):
    try:
        channel_index = channels.index(channel_name)
    except ValueError:
        print(f"Lỗi: Không tìm thấy kênh '{channel_name}'")
        return
        
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    
    axes[0].plot(train_data[:, channel_index])
    axes[0].set_title(f'Dữ liệu Train - Kênh {channel_name}', fontsize=16)
    axes[0].set_ylabel('Giá trị')
    
    axes[1].plot(test_data[:, channel_index], label='Dữ liệu Test')
    axes[1].set_title(f'Dữ liệu Test - Kênh {channel_name}', fontsize=16)
    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('Giá trị')
    
    channel_anomalies = anomaly_labels[anomaly_labels['chan_id'] == channel_name]
    if not channel_anomalies.empty:
        sequences_str = channel_anomalies['anomaly_sequences'].iloc[0]
        try:
            sequences = eval(sequences_str)
            for i, seq in enumerate(sequences):
                # Chỉ đánh dấu nếu đoạn bất thường không bị cắt mất
                if seq[1] <= test_data.shape[0]:
                    label = f"Bất thường: {channel_anomalies['attack'].iloc[0]}" if i == 0 else ""
                    axes[1].axvspan(seq[0], seq[1], color='red', alpha=0.3, label=label)
        except Exception as e:
            print(f"Lỗi khi xử lý chuỗi bất thường cho kênh {channel_name}: {e}")

    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        axes[1].legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'channel_{channel_name}_visualization.png')
    plt.savefig(save_path, dpi=600)
    print(f"Đã lưu đồ thị cho kênh {channel_name} tại {save_path}")
    plt.close()

print("\n--- Bắt đầu trực quan hóa dữ liệu ---")
plot_channel_data('P-1', train_data, test_data, anomaly_labels, CHANNELS, FIGURES_DIR)
plot_channel_data('S-1', train_data, test_data, anomaly_labels, CHANNELS, FIGURES_DIR)
plot_channel_data('E-3', train_data, test_data, anomaly_labels, CHANNELS, FIGURES_DIR)

# --- BƯỚC 2.3: CHUẨN HÓA DỮ LIỆU ---
print("\n--- Bắt đầu chuẩn hóa dữ liệu ---")
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)
print("Đã chuẩn hóa dữ liệu train và test về khoảng [0, 1].")

# --- BƯỚC 2.4: TẠO CỬA SỔ TRƯỢT ---
def create_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

print(f"\n--- Bắt đầu tạo cửa sổ trượt với kích thước w={WINDOW_SIZE} ---")
split_index = int(train_scaled.shape[0] * (1 - TRAIN_RATIO_FOR_VALIDATION))
train_only_data = train_scaled[:split_index]
validation_data = train_scaled[split_index:]

X_train, y_train = create_windows(train_only_data, WINDOW_SIZE)
X_val, y_val = create_windows(validation_data, WINDOW_SIZE)
X_test, y_test = create_windows(test_scaled, WINDOW_SIZE)

print(f"Hình dạng X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Hình dạng X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"Hình dạng X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- BƯỚC 2.5: LƯU DỮ LIỆU ĐÃ XỬ LÝ ---
print("\n--- Bắt đầu lưu dữ liệu đã xử lý ---")
np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)

with open(os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(PROCESSED_DATA_DIR, 'channels.txt'), 'w') as f:
    for channel in CHANNELS:
        f.write(f"{channel}\n")
        
print(f"Đã lưu tất cả dữ liệu đã xử lý vào thư mục '{PROCESSED_DATA_DIR}'")
print("\n--- Hoàn thành Giai đoạn 2 ---")
