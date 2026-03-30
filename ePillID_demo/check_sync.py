import pickle
import os
import sys
import sklearn.preprocessing

# VÁ LỖI TƯƠNG THÍCH PHIÊN BẢN
sys.modules['sklearn.preprocessing.label'] = sklearn.preprocessing._label

# Đường dẫn các file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(BASE_DIR, "label_encoder.pickle")
classes_path = os.path.join(BASE_DIR, "classes.txt")

def check_consistency():
    # 1. Load LabelEncoder
    if not os.path.exists(encoder_path):
        print(f"❌ Không tìm thấy file tại: {encoder_path}")
        return
    
    try:
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)
    except Exception as e:
        print(f"❌ Lỗi khi load LabelEncoder: {e}")
        return
    
    le_classes = list(le.classes_)
    print(f"✅ Số lượng nhãn trong LabelEncoder: {len(le_classes)}")

    # 2. Load classes.txt
    if not os.path.exists(classes_path):
        print(f"❌ Không tìm thấy file tại: {classes_path}")
        return
        
    with open(classes_path, "r", encoding="utf-8") as f:
        txt_classes = [line.strip() for line in f.readlines()]
    print(f"✅ Số lượng dòng trong classes.txt: {len(txt_classes)}")

    # 3. So sánh
    print("\n--- So sánh thứ tự (Index Check) ---")
    indices_to_check = [0, 1, 100, 500, len(txt_classes)-1]
    
    mismatch_count = 0
    for i in indices_to_check:
        if i >= len(le_classes) or i >= len(txt_classes):
            continue
            
        le_val = le_classes[i]
        txt_val = txt_classes[i]
        
        status = "✅ KHỚP" if le_val == txt_val else "❌ LỆCH"
        if le_val != txt_val: mismatch_count += 1
        
        print(f"Index [{i}]:")
        print(f"  - Encoder: {le_val}")
        print(f"  - TXT file: {txt_val}")
        print(f"  => Kết quả: {status}")

    if mismatch_count == 0 and len(le_classes) == len(txt_classes):
        print("\n🌟 KẾT LUẬN: Dữ liệu nhãn hoàn toàn đồng nhất!")
    else:
        print("\n⚠️ KẾT LUẬN: CÓ SỰ SAI LỆCH THỨ TỰ NHÃN.")
        print("Đây là nguyên nhân khiến Confidence thấp và dự đoán ID sai.")

if __name__ == "__main__":
    check_consistency()