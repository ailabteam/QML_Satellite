# 01_data_preprocessing.py

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

print("--- Giai đoạn 2: Khám phá và Tiền xử lý Dữ liệu ---")

# --- BƯỚC 2.1: TẢI VÀ GỘP DỮ LIỆU ---
def load_and_merge_data(path):
    all_files = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
    print(f"Tìm thấy {len(all_files)} kênh trong {path}")
    
    merged_data = []
    channel_names = []
    
    for filename in tqdm.tqdm(all_files, desc=f"Đang tải dữ liệu từ {path}"):
        channel_name = filename.replace('.npy', '')
        data = np.load(os.path.join(path, filename))
        
        # Chỉ lấy cột đầu tiên chứa giá trị telemetry thực tế
        if data.ndim > 1:
            merged_data.append(data[:, 0])
        else:
            merged_data.append(data)
        
        channel_names.append(channel_name)
        
    return np.array(merged_data).T, channel_names

train_data, train_channels = load_and_merge_data(TRAIN_DIR)
test_data, test_channels = load_and_merge_data(TEST_DIR)

print(f"Hình dạng dữ liệu train đã gộp: {train_data.shape}")
print(f"Hình dạng dữ liệu test đã gộp: {test_data.shape}")
assert train_channels == test_channels, "Các kênh trong train và test không khớp!"
CHANNELS = train_channels

anomaly_labels = pd.read_csv('labeled_anomalies.csv')
print("\nThông tin các chuỗi bất thường đã được gán nhãn:")
print(anomaly_labels.head())

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
        sequences = eval(sequences_str)
        for i, seq in enumerate(sequences):
            label = f"Bất thường: {channel_anomalies['attack'].iloc[0]}" if i == 0 else ""
            axes[1].axvspan(seq[0], seq[1], color='red', alpha=0.3, label=label)
    
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
