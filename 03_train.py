# 03_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm

# Import kiến trúc mô hình từ file trước
from model_architecture import Generator, Critic, DATA_DIM, WINDOW_SIZE, HIDDEN_DIM_GRU

print("--- Giai đoạn 4: Huấn luyện Mô hình QGRU-WGAN ---")

# --- 1. CONFIGURATION ---
PROCESSED_DATA_DIR = 'processed_data'
MODELS_DIR = 'saved_models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
N_EPOCHS = 100
N_CRITIC_STEPS = 5  # Số lần huấn luyện Critic cho mỗi lần huấn luyện Generator
LAMBDA_GP = 10      # Hệ số cho Gradient Penalty

# Device configuration (sử dụng GPU nếu có)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {DEVICE}")


# --- 2. DATA LOADING ---
print("\n--- Đang tải dữ liệu đã xử lý ---")
X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))

# Chuyển đổi sang PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Tạo TensorDataset và DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Đã tạo DataLoader với {len(train_loader)} batches, mỗi batch size {BATCH_SIZE}.")


# --- 3. MODEL, OPTIMIZER, LOSS SETUP ---
print("\n--- Khởi tạo mô hình và optimizers ---")
generator = Generator(input_dim=DATA_DIM, hidden_dim=HIDDEN_DIM_GRU, output_dim=DATA_DIM).to(DEVICE)
critic = Critic(input_dim=DATA_DIM, hidden_dim=HIDDEN_DIM_GRU).to(DEVICE)

# Optimizers
opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))


# --- 4. HELPER FUNCTIONS ---
def gradient_penalty(critic, real, fake, device):
    """Tính toán gradient penalty cho WGAN-GP."""
    BATCH_SIZE, FEATURES = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1)).repeat(1, FEATURES).to(device)
    interpolated_samples = real * epsilon + fake * (1 - epsilon)
    interpolated_samples.requires_grad_(True)

    # Tính toán score cho các sample nội suy
    interpolated_scores = critic(interpolated_samples)

    # Tính gradient
    gradient = torch.autograd.grad(
        inputs=interpolated_samples,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
        allow_unused=True  # Sửa lỗi ở đây
    )[0]

    # Xử lý trường hợp gradient là None
    if gradient is None:
        return torch.tensor(0.0).to(device)

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty



def reparameterize(mean, log_var):
    """Reparameterization trick để lấy mẫu từ phân phối Gaussian."""
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mean + eps * std


# --- 5. TRAINING LOOP ---
print("\n--- Bắt đầu vòng lặp huấn luyện ---")
for epoch in range(N_EPOCHS):
    gen_losses = []
    critic_losses = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
    for batch_idx, (real_seq, real_next_point) in enumerate(progress_bar):
        real_seq = real_seq.to(DEVICE)
        real_next_point = real_next_point.to(DEVICE)

        # --- Huấn luyện Critic ---
        for _ in range(N_CRITIC_STEPS):
            mean, log_var = generator(real_seq)
            fake_next_point = reparameterize(mean, log_var)

            critic_real = critic(real_next_point).reshape(-1)
            critic_fake = critic(fake_next_point).reshape(-1)

            gp = gradient_penalty(critic, real_next_point, fake_next_point, DEVICE)

            # Loss của Critic: muốn score của real cao, fake thấp, và thêm gradient penalty
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

            critic.zero_grad()
            loss_critic.backward(retain_graph=True) # retain_graph=True vì fake_next_point sẽ được dùng lại trong loss của Generator
            opt_critic.step()

        # --- Huấn luyện Generator ---
        # Chúng ta đã có mean, log_var từ bước trước, nhưng để tính toán gradient cho Generator
        # chúng ta cần chạy lại forward pass của critic trên dữ liệu fake mới.
        gen_fake_mean, gen_fake_log_var = generator(real_seq)
        gen_fake_sample = reparameterize(gen_fake_mean, gen_fake_log_var)

        critic_fake_for_gen = critic(gen_fake_sample).reshape(-1)

        # Loss của Generator: muốn score của dữ liệu fake cao nhất có thể
        loss_gen = -torch.mean(critic_fake_for_gen)

        # Thêm các loss phụ từ paper (KL divergence và variance penalty)
        # Đây là một bước quan trọng để ổn định và cải thiện kết quả.
        kl_div = 0.5 * torch.mean(torch.exp(gen_fake_log_var) + gen_fake_mean**2 - 1.0 - gen_fake_log_var)
        variance_penalty = torch.mean(torch.exp(gen_fake_log_var))

        # Lambda values for additional losses (can be tuned)
        LAMBDA_KL = 0.01
        LAMBDA_VAR = 0.01

        loss_gen_total = loss_gen + LAMBDA_KL * kl_div + LAMBDA_VAR * variance_penalty

        generator.zero_grad()
        loss_gen_total.backward()
        opt_gen.step()

        # Lưu lại loss
        gen_losses.append(loss_gen_total.item())
        critic_losses.append(loss_critic.item())

        # Cập nhật progress bar
        progress_bar.set_postfix(Loss_C=np.mean(critic_losses), Loss_G=np.mean(gen_losses))

    print(f"Epoch {epoch+1}/{N_EPOCHS}, Critic Loss: {np.mean(critic_losses):.4f}, Generator Loss: {np.mean(gen_losses):.4f}")

    # Lưu model checkpoint sau mỗi 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), os.path.join(MODELS_DIR, f'generator_epoch_{epoch+1}.pth'))
        torch.save(critic.state_dict(), os.path.join(MODELS_DIR, f'critic_epoch_{epoch+1}.pth'))
        print(f"Đã lưu model checkpoints tại epoch {epoch+1}")

print("\n--- Hoàn thành huấn luyện ---")
