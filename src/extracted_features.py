import sys
try:
    import sklearn.preprocessing._label as label
    sys.modules['sklearn.preprocessing.label'] = sys.modules['sklearn.preprocessing._label']
except ImportError:
    pass
import os
import torch
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import PIL
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from arguments import common_parser 
from models.enhanced_embedding_model import create_enhanced_model
from models.enhanced_multihead_model import EnhancedMultiheadModel

class EpillIDDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Sửa đường dẫn từ Linux sang Windows
        img_path_raw = self.data_frame.iloc[idx]['image_path']
        
        # Thử các chiến lược khác nhau để xây dựng đường dẫn
        candidates = []
        
        # 1. Nếu đường dẫn chứa /content/data, thay thế bằng root_dir
        if '/content/data' in str(img_path_raw):
            img_name = img_path_raw.replace('/content/data', self.root_dir).replace('/', os.sep)
            candidates.append(img_name)
        
        # 2. Thử join trực tiếp root_dir với tên file
        candidates.append(os.path.join(self.root_dir, os.path.basename(img_path_raw)))
        
        # 3. Nếu đường dẫn là tương đối, join với root_dir
        if not os.path.isabs(img_path_raw):
            clean_path = str(img_path_raw).replace('/', os.sep)
            candidates.append(os.path.join(self.root_dir, clean_path))
        
        # 4. Thử đường dẫn như là
        candidates.append(str(img_path_raw))
        
        # Tìm tập tin đầu tiên tồn tại
        img_name = None
        for candidate in candidates:
            if os.path.exists(candidate):
                img_name = candidate
                break
        
        if img_name is None:
            print(f"[ERROR] Không tìm thấy ảnh. Raw path: {img_path_raw}")
            print(f"        Các lựa chọn đã thử:")
            for cand in candidates:
                print(f"          - {cand}")
            raise FileNotFoundError(f"Image not found: {img_path_raw}")

        try:
            image = Image.open(img_name).convert('RGB')
        except (PIL.UnidentifiedImageError, IOError) as e:
            print(f"[WARNING] Không thể mở ảnh: {img_name}")
            print(f"          Lỗi: {e}")
            # Trả về ảnh đen thay vì raise error
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        label = self.data_frame.iloc[idx]['label']
        is_ref = self.data_frame.iloc[idx]['is_ref']
        
        if self.transform:
            image = self.transform(image)
        return image, label, is_ref

def run_extraction(csv_path, model_path, encoder_path, local_data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang chạy trên: {device}")

    # 1. Khởi tạo args chuẩn từ dự án
    parser = common_parser()
    args = parser.parse_args(args=[])
    
    # --- HARD OVERRIDE ARGS AT RUN_EXTRACTION ---
    print("\n" + "!"*60)
    print("!!! HARD OVERRIDE ARGS AT RUN_EXTRACTION !!!")
    args.pooling = 'GAvP'
    args.metric_embedding_dim = 2048
    # Nếu args có các thuộc tính này thì ép luôn
    if hasattr(args, 'use_coord_attention'): args.use_coord_attention = True
    if hasattr(args, 'use_domain_adaptation'): args.use_domain_adaptation = True
    print(f"[*] Pooling set to: {args.pooling}")
    print(f"[*] Embedding Dim set to: {args.metric_embedding_dim}")
    print("!"*60 + "\n")
    # --------------------------------- 
    
    # CÀI ĐẶT ĐỂ KHỚP VỚI CHECKPOINT (Tránh lỗi 2048 vs 256)
    args.appearance_network = 'resnet50'
    args.pooling = 'GAvP'           # PHẢI LÀ GAvP để có đầu ra 2048
    args.metric_embedding_dim = 2048 # Khớp với shape trong checkpoint
    args.dropout = 0.5
    
    # 2. Load Label Encoder
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
        num_classes = len(le.classes_)
    args.num_classes = num_classes

    # 3. Khởi tạo Model 
    # Nếu nghiên cứu sự ảnh hưởng, use_coord_attention phải khớp với lúc bạn TRAIN file .pth này
    e_model = create_enhanced_model(
        args=args, 
        use_coord_attention=True,   # Đổi thành False nếu checkpoint này chưa có CA
        use_domain_adaptation=True  # Đổi thành False nếu checkpoint này chưa có GRL
    )
    
    # Khởi tạo Multihead (Sửa lỗi missing num_classes)
    model = EnhancedMultiheadModel(e_model, n_classes=num_classes)
    
    # LOAD TRỌNG SỐ - Chỉ load những layer có shape khớp
    print(f"[*] Đang load trọng số từ {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    current_state_dict = model.state_dict()
    
    # Lọc lấy chỉ những key có shape khớp nhau
    filtered_checkpoint = {}
    for k, v in checkpoint.items():
        if k in current_state_dict:
            if v.shape == current_state_dict[k].shape:
                filtered_checkpoint[k] = v
            else:
                print(f"[!] Shape mismatch for {k}: checkpoint {v.shape} vs model {current_state_dict[k].shape}, skipping...")
        else:
            print(f"[!] Key {k} not in current model, skipping...")
    
    model.load_state_dict(filtered_checkpoint, strict=False)
    model.to(device)
    model.eval()

    # 4. DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = EpillIDDataset(csv_file=csv_path, root_dir=local_data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    features_list, labels_list, domains_list = [], [], []

    with torch.no_grad():
        for batch_data in dataloader:
            # Handle both tuple and dict returns from dataloader
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                inputs, labels, is_ref = batch_data
            else:
                inputs = batch_data[0]
                labels = batch_data[1]
                is_ref = batch_data[2]
            
            # Ensure inputs is a tensor
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs)
            
            # Convert labels: handle different input types
            if isinstance(labels, torch.Tensor):
                # Already a tensor, check dtype
                if labels.dtype != torch.long:
                    labels = labels.long()
            else:
                # Convert to list if tuple, then use label encoder
                labels_list = list(labels) if isinstance(labels, tuple) else labels
                try:
                    labels_numeric = le.transform(labels_list)
                    labels = torch.tensor(labels_numeric, dtype=torch.long)
                except Exception as e:
                    print(f"[ERROR] Could not transform labels: {e}")
                    print(f"        Label type: {type(labels_list)}, first element: {labels_list[0]}")
                    raise
            
            # Convert is_ref to tensor
            if not isinstance(is_ref, torch.Tensor):
                is_ref_list = list(is_ref) if isinstance(is_ref, tuple) else is_ref
                is_ref = torch.tensor(is_ref_list)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Model forward pass - use try/except to catch dimension mismatch
            try:
                outputs = model(inputs, labels)
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    # Dimension mismatch - try embedding model directly
                    print(f"[WARNING] Dimension mismatch, using embedding model directly: {e}")
                    emb = e_model(inputs)
                    if isinstance(emb, tuple):
                        emb = emb[0]
                    outputs = {'embeddings': emb}
                else:
                    raise
            
            # Handle both dict and tuple outputs
            if isinstance(outputs, dict):
                emb = outputs['embeddings']
            elif isinstance(outputs, (tuple, list)):
                # If tuple/list, assume first element is embeddings
                emb = outputs[0]
            else:
                # If single tensor, use it directly
                emb = outputs
            
            features_list.append(emb.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            domains_list.append(is_ref.cpu().numpy())

    # 5. Lưu kết quả
    np.save('features.npy', np.concatenate(features_list, axis=0))
    np.save('labels.npy', np.concatenate(labels_list, axis=0))
    np.save('domains.npy', np.concatenate(domains_list, axis=0))
    print(f"[V] Đã trích xuất xong {len(dataset)} mẫu. File đã sẵn sàng để phân tích tương quan!")

if __name__ == "__main__":
    # ĐƯỜNG DẪN CỤC BỘ TRÊN WINDOWS CỦA BẠN
    DATA_PATH = r'C:\Users\Huynhtu\OneDrive - VLG\Desktop\ePillID-benchmark-master\ePillID-benchmark-master\src\ePillID_data\classification_data'
    
    run_extraction(
        csv_path='all_labels.csv', 
        model_path='best_model.pth', 
        encoder_path='label_encoder.pickle',
        local_data_dir=DATA_PATH
    )