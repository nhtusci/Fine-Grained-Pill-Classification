import torch

# Đường dẫn tới file model của bạn
checkpoint_path = "best_model.pth"

try:
    # 1. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 2. Xác định state_dict (đề phòng trường hợp file lưu cả epoch, optimizer)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("--- Thông tin Checkpoint ---")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best Accuracy: {checkpoint.get('best_acc', 'N/A')}")
    else:
        state_dict = checkpoint

    print(f"\n--- Danh sách 10 Layer đầu tiên trong file .pth ---")
    keys = list(state_dict.keys())
    for key in keys[:10]:
        print(f"Key: {key} | Kích thước: {state_dict[key].shape}")

    # 3. Kiểm tra tiền tố 'module.'
    has_module = any(k.startswith('module.') for k in keys)
    print(f"\nCó chứa tiền tố 'module.' không: {has_module}")

    # 4. Kiểm tra tiền tố 'embedding_model.'
    has_embedding = any(k.startswith('embedding_model.') for k in keys)
    print(f"Có chứa 'embedding_model.' không: {has_embedding}")

except Exception as e:
    print(f"Lỗi: {e}")