# 02_model_architecture.py (v3 - Sửa lỗi ValueError shape và cách gọi Layer)

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

print("--- Giai đoạn 3: Xây dựng Kiến trúc Mô hình (v3) ---")

# --- CONFIGURATION ---
N_QUBITS = 4
N_LAYERS = 2
DATA_DIM = 82
WINDOW_SIZE = 5
HIDDEN_DIM_GRU = 32

# --- BƯỚC 3.1: HYBRID QUANTUM LAYER (HQL) ---
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(inputs, weights):
    """
    weights ở đây là tensor trọng số cho TẤT CẢ các lớp.
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    # === SỬA LỖI Ở ĐÂY: Gọi Layer một lần duy nhất ===
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    # ===============================================
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class HQLayer(nn.Module):
    def __init__(self, in_features, out_features, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        
        self.classical_layer = nn.Linear(in_features, n_qubits)
        
        # Cách khởi tạo trọng số này là đúng, giữ nguyên
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=self.n_qubits)
        initial_weights = torch.rand(shape) * 2 * torch.pi
        self.quantum_weights = nn.Parameter(initial_weights)

        self.classical_output = nn.Linear(n_qubits, out_features)

    def forward(self, x):
        x = self.classical_layer(x)
        
        quantum_results = []
        for x_single in x:
            # === SỬA LỖI NHỎ Ở ĐÂY: Truyền thẳng self.quantum_weights, không cần .double() ===
            q_out_tensor = torch.tensor(quantum_circuit(x_single.double(), self.quantum_weights), dtype=x.dtype)
            quantum_results.append(q_out_tensor)
        
        x = torch.stack(quantum_results)
        x = x.to(self.classical_output.weight.device)

        x = self.classical_output(x)
        
        return x

# --- Các lớp QGRUCell, Generator, Critic không thay đổi ---
class QGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.hql_update = HQLayer(in_features=input_dim + hidden_dim, out_features=hidden_dim)
        self.hql_reset = HQLayer(in_features=input_dim + hidden_dim, out_features=hidden_dim)
        self.hql_candidate = HQLayer(in_features=input_dim + hidden_dim, out_features=hidden_dim)

    def forward(self, x_t, h_prev):
        combined_input = torch.cat([x_t, h_prev], dim=1)
        z_t = torch.sigmoid(self.hql_update(combined_input))
        r_t = torch.sigmoid(self.hql_reset(combined_input))
        candidate_input = torch.cat([x_t, r_t * h_prev], dim=1)
        h_tilde_t = torch.tanh(self.hql_candidate(candidate_input))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
        return h_t

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.qgru_cell = QGRUCell(input_dim=input_dim, hidden_dim=hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_log_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            h_t = self.qgru_cell(x_t, h_t)
        mean = self.fc_mean(h_t)
        log_var = self.fc_log_var(h_t)
        return mean, log_var

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.qgru_cell = QGRUCell(input_dim=input_dim, hidden_dim=hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        h_t = self.qgru_cell(x, h_t)
        score = self.fc_out(h_t)
        return score

print("Đã định nghĩa xong các lớp HQLayer, QGRUCell, Generator, và Critic.")

# --- KIỂM TRA NHANH KIẾN TRÚC ---
def test_architecture():
    print("\n--- Bắt đầu kiểm tra nhanh kiến trúc ---")
    batch_size = 4
    
    test_input_sequence = torch.randn(batch_size, WINDOW_SIZE, DATA_DIM)
    test_single_point = torch.randn(batch_size, DATA_DIM)

    generator = Generator(input_dim=DATA_DIM, hidden_dim=HIDDEN_DIM_GRU, output_dim=DATA_DIM)
    critic = Critic(input_dim=DATA_DIM, hidden_dim=HIDDEN_DIM_GRU)

    print(f"Generator: {generator}")
    print(f"Critic: {critic}")

    print("\nTesting Generator...")
    mean, log_var = generator(test_input_sequence)
    print(f"Input shape (Generator): {test_input_sequence.shape}")
    print(f"Output mean shape: {mean.shape}")
    print(f"Output log_var shape: {log_var.shape}")
    assert mean.shape == (batch_size, DATA_DIM)
    assert log_var.shape == (batch_size, DATA_DIM)
    print("Generator test PASSED.")

    print("\nTesting Critic...")
    score = critic(test_single_point)
    print(f"Input shape (Critic): {test_single_point.shape}")
    print(f"Output score shape: {score.shape}")
    assert score.shape == (batch_size, 1)
    print("Critic test PASSED.")
    
    print("\n--- Kiến trúc mô hình hoạt động chính xác! ---")

if __name__ == '__main__':
    test_architecture()
