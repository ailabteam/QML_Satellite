import numpy as np
import sys

if len(sys.argv) != 2:
    print("Sử dụng: python explore_channel.py <đường_dẫn_tới_file.npy>")
    sys.exit(1)

file_path = sys.argv[1]
try:
    data = np.load(file_path)
    print(f"Thông tin file: {file_path}")
    print(f"  - Kiểu dữ liệu (dtype): {data.dtype}")
    print(f"  - Số chiều (ndim): {data.ndim}")
    print(f"  - Kích thước (shape): {data.shape}")
    if data.size > 0:
        print(f"  - Giá trị min: {np.min(data)}")
        print(f"  - Giá trị max: {np.max(data)}")
        print(f"  - Giá trị trung bình: {np.mean(data)}")
        print("\n5 giá trị đầu tiên:")
        print(data[:5])
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại {file_path}")
