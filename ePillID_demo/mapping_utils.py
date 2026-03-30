import json
import os
import requests
import re
import socket

# Ép hệ thống dùng IPv4 để tránh lỗi kết nối một số API
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = getaddrinfo_ipv4

class DrugMapper:
    def __init__(self, classes_path="classes.txt", dict_path="drug_dict.json"):
        self.dict_path = dict_path
        # 1. Nạp danh sách Class (Index -> ID)
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"⚠️ Lỗi nạp classes.txt: {e}")
            self.class_names = []
        
        # 2. Nạp Từ điển (ID -> Thông tin chi tiết)
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    self.drug_dict = json.load(f)
            except:
                self.drug_dict = {}
        else:
            self.drug_dict = {}

    def _save_dict(self):
        """Lưu lại từ điển offline sau khi cập nhật dữ liệu từ NIH"""
        try:
            with open(self.dict_path, 'w', encoding='utf-8') as f:
                json.dump(self.drug_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Không thể lưu từ điển: {e}")

    def _clean_brand_name(self, concept_name):
        """
        Bóc tách Brand Name từ chuỗi NIH. 
        Ví dụ: '{...} [LoSeasonique]' -> 'LoSeasonique'
        Ví dụ: 'Simvastatin 80 MG Oral Tablet' -> 'Simvastatin'
        """
        if not concept_name: return "Unknown"
        
        # Bước 1: Tìm tên trong ngoặc vuông [] trước (Thường là tên thương mại chuẩn nhất từ NIH)
        match_bracket = re.search(r'\[(.*?)\]', concept_name)
        if match_bracket:
            return match_bracket.group(1).strip()
            
        # Bước 2: Loại bỏ phần bắt đầu bằng dấu ngoặc nhọn hoặc tròn rác
        clean_name = re.sub(r'^[\{\(\s]+', '', concept_name)

        # Bước 3: Chỉ lấy phần chữ trước khi gặp con số (nồng độ thuốc)
        # Regex này lấy các ký tự chữ cái và khoảng trắng ở đầu chuỗi
        match = re.search(r'^([a-zA-Z\s\-]+)', clean_name)
        if match:
            clean_name = match.group(1).strip()
            
        # Bước 4: Loại bỏ các từ khóa về dạng bào chế dư thừa ở cuối để tránh lỗi OpenFDA
        clean_name = re.sub(r'\s+(Oral|Tablet|Capsule|Gel|Cream|Solution|Pack|Injection|Suspension)$', '', clean_name, flags=re.IGNORECASE)
        
        return clean_name.title()

    def fetch_from_nih(self, ndc_code):
        """Truy vấn trực tiếp đến API của NIH (RxNav)"""
        clean_ndc = str(ndc_code).split('_')[0]
        url = f"https://rxnav.nlm.nih.gov/REST/ndcstatus.json?ndc={clean_ndc}"
        
        try:
            print(f"🌐 [NIH] Đang tra cứu trực tuyến mã: {clean_ndc}")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                ndc_status = data.get("ndcStatus", {})
                concept_name = ndc_status.get("conceptName")
                
                if concept_name and concept_name != "Unknown":
                    # Tách tên brand thuần túy để tra cứu tiếp ở OpenFDA
                    brand_only = self._clean_brand_name(concept_name)
                    print(f"✅ [NIH] Trả về: {concept_name} -> Lọc: {brand_only}")
                    return {
                        "brand_name": brand_only,
                        "generic_name": concept_name,
                        "manufacturer": "NIH RxNav Online"
                    }
        except Exception as e:
            print(f"⚠️ [NIH] Lỗi kết nối: {str(e)}")
        return None

    def get_drug_info(self, class_index):
        """Hàm xử lý chính: Online (NIH) -> Update Offline -> Fallback Offline"""
        try:
            pill_id = self.class_names[class_index]
            prefix = pill_id.split('_')[0]
            parts = prefix.split('-')
            # Chuẩn hóa key lưu trữ (VD: 00093-7156)
            clean_id = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else prefix

            # --- BƯỚC 1: TRA CỨU NIH TRƯỚC ---
            online_data = self.fetch_from_nih(prefix)

            if online_data:
                # Cập nhật thông tin vào từ điển offline
                self.drug_dict[clean_id] = online_data
                self._save_dict()
                return {
                    "pill_id": pill_id,
                    **online_data
                }

            # --- BƯỚC 2: NẾU NIH THẤT BẠI, KIỂM TRA OFFLINE ---
            print(f"🔍 [Offline] Kiểm tra từ điển cho: {clean_id}")
            info = self.drug_dict.get(clean_id)
            
            if info:
                return { "pill_id": pill_id, **info }

            return {
                "pill_id": pill_id,
                "brand_name": f"Unknown ({clean_id})",
                "generic_name": "N/A",
                "manufacturer": "N/A"
            }
        except Exception as e:
            print(f"⚠️ Lỗi get_drug_info: {e}")
            return None

    def smart_summarize(self, text):
        if not text or text == "N/A": return "Không có dữ liệu."
        sentences = text.split('.')
        short_summary = ". ".join(sentences[:2]).strip()
        return (short_summary[:147] + "...") if len(short_summary) > 150 else (short_summary + ".")

    def fetch_fda_online(self, brand_name):
        """Dùng brand_name đã được lọc sạch để tra OpenFDA"""
        if not brand_name or "Unknown" in brand_name:
            return self._get_empty_fda_data("Không có tên thuốc hợp lệ.")
        try:
            # Dùng dấu ngoặc kép quanh brand_name để tìm chính xác cụm từ
            url = f'https://api.fda.gov/drug/label.json?search=openfda.brand_name:"{brand_name}"&limit=1'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) > 0:
                    res = data["results"][0]
                    usage = res.get("indications_and_usage", ["N/A"])[0]
                    dosage = res.get("dosage_and_administration", ["N/A"])[0]
                    warn = res.get("warnings", res.get("warnings_and_cautions", ["N/A"]))[0]
                    return {
                        "summary": {
                            "usage": self.smart_summarize(usage),
                            "dosage": self.smart_summarize(dosage),
                            "warnings": self.smart_summarize(warn)
                        },
                        "full_text": {"usage": usage, "dosage": dosage, "warnings": warn},
                        "source": "OpenFDA"
                    }
            return self._get_empty_fda_data(f"Không tìm thấy thông tin cho '{brand_name}' trên OpenFDA.")
        except Exception as e:
            return self._get_empty_fda_data(f"Lỗi kết nối FDA: {str(e)}")

    def _get_empty_fda_data(self, message):
        return {
            "summary": {"usage": message, "dosage": "N/A", "warnings": "N/A"},
            "full_text": {"usage": message, "dosage": "N/A", "warnings": "N/A"},
            "source": "None"
        }