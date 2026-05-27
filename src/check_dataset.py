import os

train_dir = "data/train"
val_dir = "data/val"

print("\n--- TRAIN SET ---")
for cls in os.listdir(train_dir):
    path = os.path.join(train_dir, cls)
    if os.path.isdir(path):
        print(cls, ":", len(os.listdir(path)))

print("\n--- VAL SET ---")
for cls in os.listdir(val_dir):
    path = os.path.join(val_dir, cls)
    if os.path.isdir(path):
        print(cls, ":", len(os.listdir(path)))