# 04_inference_and_evaluation.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

# Import kiến trúc mô hình và các hằng số
from model_architecture import Generator, Critic, DATA_DIM, WINDOW_SIZE, HIDDEN_DIM_GRU

print("--- Giai đoạn 5: Đánh giá và Phát hiện Bất thường ---")

# --- 1. CONFIGURATION ---
PROCESSED_DATA_DIR = 'processed_data'
MODELS_DIR = 'saved_models'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Chọn epoch của model để tải. Ví dụ: 10, 20, ..., 100
MODEL_EPOCH = 10 
GENERATOR_MODEL_PATH = os.path.join(MODELS_DIR, f'generator_epoch_{MODEL_EPOCH}.pth')
CRITIC_MODEL_PATH = os.path.join(MODELS_DIR, f'critic_epoch_{MODEL_EPOCH}.pth')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {DEVICE}")


# --- 2. TẢI DỮ LIỆU VÀ MODEL ---
print("\n--- Đang tải dữ liệu test, scaler, và models ---")
X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))

# Tải scaler để biến đổi ngược dữ liệu về thang đo gốc (nếu cần)
with open(os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Tải danh sách kênh
CHANNELS = [line.strip() for line in open(os.path.join(PROCESSED_DATA_DIR, 'channels.txt'))]

# Tải nhãn bất thường
anomaly_labels = pd.read_csv('labeled_anomalies.csv')

# Khởi tạo và tải trọng số cho model
generator = Generator(input_dim=DATA_DIM, hidden_dim=HIDDEN_DIM_GRU, output_dim=DATA_DIM).to(DEVICE)
critic = Critic(input_dim=DATA_DIM, hidden_dim=HIDDEN_DIM_GRU).to(DEVICE)

generator.load_state_dict(torch.load(GENERATOR_MODEL_PATH))
critic.load_state_dict(torch.load(CRITIC_MODEL_PATH))

# Chuyển model sang chế độ đánh giá (rất quan trọng!)
generator.eval()
critic.eval()

print(f"Đã tải thành công models từ epoch {MODEL_EPOCH}.")


# --- 3. CHẠY INFERENCE VÀ TÍNH ĐIỂM BẤT THƯỜNG ---
print("\n--- Bắt đầu chạy inference trên tập test ---")

# Chuyển dữ liệu test sang tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

# Lists để lưu kết quả
all_means = []
all_log_vars = []
all_critic_scores = []

with torch.no_grad(): # Không cần tính gradient trong lúc inference
    for i in tqdm(range(len(X_test_tensor)), desc="Inference"):
        seq = X_test_tensor[i:i+1] # Lấy từng chuỗi một để giữ batch size là 1
        
        # Lấy dự đoán từ Generator
        mean, log_var = generator(seq)
        all_means.append(mean.cpu().numpy())
        all_log_vars.append(log_var.cpu().numpy())
        
        # Lấy điểm từ Critic cho điểm dữ liệu thật tiếp theo
        real_point = torch.tensor(y_test[i:i+1], dtype=torch.float32).to(DEVICE)
        critic_score = critic(real_point)
        all_critic_scores.append(critic_score.cpu().numpy())

# Chuyển đổi list thành mảng numpy
predicted_means = np.concatenate(all_means)
predicted_log_vars = np.concatenate(all_log_vars)
critic_scores = np.concatenate(all_critic_scores).flatten()

# --- 4. TÍNH TOÁN ĐIỂM BẤT THƯỜNG TỔNG HỢP ---
# Paper gốc có chiến lược Gating, chúng ta sẽ implement phiên bản đơn giản trước:
# Anomaly Score = w1 * Reconstruction Error + w2 * Critic Score + w3 * Uncertainty

# 4.1. Reconstruction Error (dựa trên mean)
reconstruction_error = np.mean((y_test - predicted_means)**2, axis=1)

# 4.2. Uncertainty (dựa trên variance)
uncertainty = np.mean(np.exp(predicted_log_vars), axis=1)

# 4.3. Critic Score (chúng ta muốn score thấp là bất thường)
# Do đó, chúng ta có thể dùng -critic_score
critic_component = -critic_scores

# Chuẩn hóa các thành phần về cùng một thang đo [0, 1] để kết hợp
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

re_scaled = min_max_scale(reconstruction_error)
unc_scaled = min_max_scale(uncertainty)
crit_scaled = min_max_scale(critic_component)

# Kết hợp lại (có thể điều chỉnh trọng số w1, w2, w3)
w1, w2, w3 = 0.5, 0.3, 0.2 
anomaly_scores = w1 * re_scaled + w2 * crit_scaled + w3 * unc_scaled

print("\n--- Đã tính toán xong điểm bất thường cho toàn bộ tập test ---")
print(f"Shape của Anomaly Scores: {anomaly_scores.shape}")


# --- 5. TRỰC QUAN HÓA KẾT QUẢ ---
def plot_results(channel_name, y_test, anomaly_scores, anomaly_labels, channels, scaler, save_dir):
    try:
        channel_index = channels.index(channel_name)
    except ValueError:
        print(f"Lỗi: Không tìm thấy kênh '{channel_name}'")
        return
        
    fig, ax1 = plt.subplots(figsize=(20, 8))
    
    # Dữ liệu gốc (đã biến đổi ngược)
    original_data = scaler.inverse_transform(y_test)
    
    # Trục y bên trái cho dữ liệu gốc
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Original Value', color='tab:blue')
    ax1.plot(original_data[:, channel_index], color='tab:blue', label='Ground Truth')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Trục y bên phải cho điểm bất thường
    ax2 = ax1.twinx()
    ax2.set_ylabel('Anomaly Score', color='tab:orange')
    ax2.plot(anomaly_scores, color='tab:orange', alpha=0.8, label='Anomaly Score')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    # Đánh dấu các vùng bất thường ground-truth
    channel_anomalies = anomaly_labels[anomaly_labels['chan_id'] == channel_name]
    if not channel_anomalies.empty:
        sequences_str = channel_anomalies['anomaly_sequences'].iloc[0]
        try:
            sequences = eval(sequences_str)
            for i, seq in enumerate(sequences):
                label = 'True Anomaly' if i == 0 else ""
                ax1.axvspan(seq[0], seq[1], color='red', alpha=0.2, label=label)
        except Exception:
            pass # Bỏ qua nếu có lỗi eval

    fig.suptitle(f'Anomaly Detection Results for Channel: {channel_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Gộp legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    save_path = os.path.join(save_dir, f'results_channel_{channel_name}_epoch{MODEL_EPOCH}.png')
    plt.savefig(save_path, dpi=600)
    print(f"Đã lưu đồ thị kết quả cho kênh {channel_name} tại {save_path}")
    plt.close()

print("\n--- Bắt đầu trực quan hóa kết quả ---")
# Vẽ cho các kênh tương tự
plot_results('P-1', y_test, anomaly_scores, anomaly_labels, CHANNELS, scaler, RESULTS_DIR)
plot_results('S-1', y_test, anomaly_scores, anomaly_labels, CHANNELS, scaler, RESULTS_DIR)
plot_results('E-3', y_test, anomaly_scores, anomaly_labels, CHANNELS, scaler, RESULTS_DIR)

print("\n--- Hoàn thành Giai đoạn 5 ---")
