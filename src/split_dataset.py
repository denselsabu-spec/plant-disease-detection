import os
import random
import shutil

#original dataset path
source_dir = "data/PlantVillage"

#Destination folders
train_dir = "data/train"
val_dir = "data/val"

#train-validation split ration
split_ratio = 0.8

#Create train and val directories if they dont exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

#loop through each class folder
for class_name in os.listdir(source_dir):

    class_path = os.path.join(source_dir, class_name)

    #skip files that are not folders
    if not os.path.isdir(class_path):
        continue
    
    #get all image filenames
    valid_extensions = (".jpg",".jpeg",".png")

    images =[]

    for item in os.listdir(class_path):
        item_path=os.path.join(class_path,item)

        if os.path.isfile(item_path) and item.lower().endswith(valid_extensions):
            images.append(item)

    #shuffle images randomly
    random.shuffle(images)

    #split index
    split_index = int(len(images)*split_ratio)

    #split into train and validation
    train_images = images[:split_index]
    val_images = images[split_index:]

    #Create claass folders inside train and val
    os.makedirs(os.path.join(train_dir,class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    #copy training images
    for image in train_images:
        src_path = os.path.join(class_path, image)

        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(train_dir,class_name,image)
        shutil.copy(src_path, dst_path)

    #copy validation images
    for image in val_images:
        src_path = os.path.join(class_path,image)
        
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(val_dir,class_name,image)
        shutil.copy(src_path, dst_path)

