import torch

checkpoint = torch.load("best_model.pth", map_location="cpu")
all_keys = list(checkpoint.keys())

print(f"Tổng số keys: {len(all_keys)}")
print("\n--- Keys cuối (20 keys) ---")
print(all_keys[-20:])

print("\n--- Tìm coord_attention ---")
ca_keys = [k for k in all_keys if 'coord' in k]
print(ca_keys if ca_keys else "KHÔNG có coord_attention")

print("\n--- Tìm domain ---")
domain_keys = [k for k in all_keys if 'domain' in k]
print(domain_keys if domain_keys else "KHÔNG có domain")

print("\n--- Tìm emb (embedding layer) ---")
emb_keys = [k for k in all_keys if 'emb.' in k]
print(emb_keys)