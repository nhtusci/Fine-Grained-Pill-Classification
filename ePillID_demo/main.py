from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import pickle
from PIL import Image
import io
import sys
import os
import torchvision.transforms as transforms
import sklearn.preprocessing
import json

# ==========================================
# VÁ LỖI TƯƠNG THÍCH
# ==========================================
sys.modules['sklearn.preprocessing.label'] = sklearn.preprocessing._label

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & DEVICE
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, 'src')
MODELS_PATH = os.path.join(SRC_PATH, 'models')
FAST_MPN_PATH = os.path.join(MODELS_PATH, 'fast-MPN-COV')

for p in [SRC_PATH, MODELS_PATH, FAST_MPN_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.enhanced_embedding_model import EnhancedEmbeddingModel
from models.enhanced_multihead_model import EnhancedMultiheadModel
from mapping_utils import DrugMapper

app = FastAPI(title="EpillID Optimized API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. KHỞI TẠO MAPPING
# ==========================================
mapper = DrugMapper(
    classes_path=os.path.join(BASE_DIR, "classes.txt"),
    dict_path=os.path.join(BASE_DIR, "drug_dict.json")
)

# ==========================================
# 3. LOAD MODEL & TRỌNG SỐ (CẬP NHẬT)
# ==========================================
print(f"🚀 Đang khởi động Model trên: {device}")

# Load Label Encoder
label_encoder_path = os.path.join(BASE_DIR, "label_encoder.pickle")
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

n_classes = len(label_encoder.classes_)

# Khởi tạo kiến trúc Model
E_model = EnhancedEmbeddingModel(
    network='resnet50', pooling='GAvP', dropout_p=0.0, cont_dims=2048,
    pretrained=False, use_coord_attention=True, use_domain_adaptation=True,
    ca_reduction=32, domain_hidden_dim=256, domain_dropout=0.0
)

model = EnhancedMultiheadModel(
    embedding_model=E_model, n_classes=n_classes,
    train_with_side_labels=True, return_domain_logits=False
)

# --- LOGIC NẠP TRỌNG SỐ TỐI ƯU ---
checkpoint_path = os.path.join(BASE_DIR, "best_model.pth")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    
    # Xử lý làm sạch Key (Xóa module. nếu có)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') # Xóa tiền tố Parallel
        new_state_dict[name] = v
    
    # Nạp với strict=False để khớp tối đa các layer hiện có
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Kết quả nạp Weight: {msg}")
else:
    print("❌ CẢNH BÁO: Không tìm thấy file best_model.pth!")

model.to(device)
model.eval()

# Transform chuẩn (Cần khớp với lúc Train)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. API ENDPOINT (ĐÃ THÊM NGƯỠNG TIN CẬY)
# ==========================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Đọc và tiền xử lý ảnh
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        # 2. AI dự đoán
        with torch.no_grad():
            logits = model.get_original_logits(tensor)
            probs = torch.softmax(logits, dim=1)
            prob, idx = torch.max(probs, dim=1)

        confidence_value = prob.item() * 100
        confidence = round(confidence_value, 2)

        # 3. KIỂM TRA NGƯỠNG TIN CẬY (THRESHOLD)
        if confidence_value < 40.0:
            return {
                "status": "low_confidence",
                "message": "System failed to identify any medication in the provided image. Please capture a clear image of the pill on a solid background.",
                "confidence": f"{confidence}%",
                "prediction": None
            }

        # 4. LẤY THÔNG TIN 
        offline_data = mapper.get_drug_info(idx.item())
        if not offline_data:
            return {"status": "error", "message": "Index không tồn tại trong danh sách nhãn."}

        brand_name = offline_data.get("brand_name", "")
        
        # Xử lý trường hợp không nhận diện được thuốc
        if "UNKNOWN" in brand_name.upper():
            return {
                "status": "not_pill",
                "pill_info": {"pill_id": "N/A", "brand_name": "Unidentified Object"},
                "display_quick": {
                    "usage": "No recognizable pill detected in this image.",
                    "dosage": "N/A",
                    "warnings": "Please ensure the pill is clearly visible on a solid background."
                },
                "display_full": None
            }

        # --- PHẦN SỬA LỖI KẾT NỐI FDA ---
        try:
            # Gọi API online từ mapper của bạn
            fda_data = mapper.fetch_fda_online(brand_name)
            
            # Kiểm tra nếu fda_data chứa lỗi kết nối (dựa trên ảnh image_d72227.jpg của bạn)
            summary = fda_data.get("summary", {})
            usage_text = summary.get("usage", "")
            
            if "HTTPSConnectionPool" in usage_text or "Lỗi kết nối" in usage_text:
                raise Exception("External API Timeout") # Ép vào nhánh catch bên dưới

        except Exception as fda_error:
            # Nếu FDA lỗi, trả về data offline hoặc thông báo thân thiện
            print(f"⚠️ FDA API Error: {fda_error}")
            return {
                "status": "success",
                "pill_info": {
                    "pill_id": offline_data["pill_id"],
                    "brand_name": brand_name
                },
                "display_quick": {
                    "usage": "Detailed information is currently unavailable due to network issues. Please try again later.",
                    "dosage": "Please consult a healthcare professional.",
                    "warnings": "Ensure the pill matches the identification before use."
                },
                "display_full": None
            }

        # Nếu mọi thứ thành công
        return {
            "status": "success",
            "pill_info": {
                "pill_id": offline_data["pill_id"],
                "brand_name": brand_name
            },
            "display_quick": fda_data["summary"],
            "display_full": fda_data["full_text"]
        }

    except Exception as e:
        return {"status": "error", "message": f"Server Error: {str(e)}"}