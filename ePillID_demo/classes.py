import sklearn.preprocessing
import sys

# Fix tương thích sklearn
sys.modules['sklearn.preprocessing.label'] = sklearn.preprocessing._label
import pickle

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# Ghi lại classes.txt đúng thứ tự label_encoder đã học
with open("classes.txt", "w") as f:
    for cls in label_encoder.classes_:
        f.write(cls + "\n")

print("✅ classes.txt đã được sync với label_encoder!")