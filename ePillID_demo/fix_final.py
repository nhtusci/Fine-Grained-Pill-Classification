import os

model_init_path = r"src\models\fast-MPN-COV\model_init.py"

with open(model_init_path, "r") as f:
    content = f.read()

# Thay dòng import cũ bằng import trực tiếp từ đúng file
old_import = "from src.network import *"
new_import = """import os as _os, sys as _sys, importlib.util as _util
_cov_dir = _os.path.dirname(_os.path.abspath(__file__))
_spec = _util.spec_from_file_location("_cov_network", _os.path.join(_cov_dir, "src", "network.py"))
_mod = _util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
for _name in dir(_mod):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_mod, _name)
del _os, _sys, _util, _spec, _mod, _name"""

content = content.replace(old_import, new_import)

with open(model_init_path, "w") as f:
    f.write(content)

print("✅ Đã sửa xong model_init.py!")