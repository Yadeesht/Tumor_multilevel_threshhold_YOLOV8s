import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import random

source_dir = "C:\\Users\\Yadeesh\\Downloads\\Brain Tumor MRI Image Dataset\\Data"
yolo_base_dir = "C:\\Users\\Yadeesh\\Downloads\\Tumor\\YOLO"

class_map ={'glioma': 0, 'meningioma': 1, 'pituitary tumor': 2,'no tumor':3}

if os.path.exists(yolo_base_dir):
    shutil.rmtree(yolo_base_dir)

os.makedirs(os.path.join(yolo_base_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(yolo_base_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(yolo_base_dir, 'val/images'), exist_ok=True)
os.makedirs(os.path.join(yolo_base_dir, 'val/labels'), exist_ok=True)

from skimage.filters import threshold_multiotsu

def generate_yolo_label(img, label_path, class_index):
    if img is None:
        print(f"Warning: Could not read image for labeling.")
        return False

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        thresholds = threshold_multiotsu(gray_img, classes=4)
        regions = np.digitize(gray_img, bins=thresholds)
    except ValueError as e:
        print(f"Warning: Multi-otsu thresholding failed: {e}")
        return False

    tumor_mask = np.where(regions == 2, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False

    h, w = gray_img.shape

    yolo_labels = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, box_w, box_h = cv2.boundingRect(contour)

            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            norm_w = box_w / w
            norm_h = box_h / h

            yolo_labels.append(f"{class_index} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    if not yolo_labels:
        return False

    with open(label_path, 'w') as f:
        for label in yolo_labels:
            f.write(label + '\n')

    return True

all_image_paths = []
for class_name, class_idx in class_map.items():
    class_dir = os.path.join(source_dir, class_name)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append((os.path.join(class_dir, filename), class_idx))

train_files, val_files = train_test_split(all_image_paths, test_size=0.2, random_state=42, stratify=[item[1] for item in all_image_paths])

def process_files(file_list, subset):
    print(f"\nProcessing {subset} set...")
    # i=0
    for img_path, class_idx in file_list:
        try:
            # i+=1
            # print(i)
            filename = os.path.basename(img_path)

            new_img_path = os.path.join(yolo_base_dir, subset, 'images', filename)
            label_path = os.path.join(yolo_base_dir, subset, 'labels', os.path.splitext(filename)[0] + '.txt')

            img = cv2.imread(img_path)

            if generate_yolo_label(img, label_path, class_idx):
                shutil.copy(img_path, new_img_path)
        except Exception as e:
            print(f"Could not process {img_path}: {e}")
process_files(train_files, 'train')
process_files(val_files, 'val')

yolo_config = {
    'path': yolo_base_dir, 
    'train': 'train/images',  
    'val': 'val/images',      
    'nc': len(class_map),     
    'names': list(class_map.keys()) 
}

with open('C:\\Users\\Yadeesh\\Downloads\\Tumor\\data.yaml', 'w') as f:
    yaml.dump(yolo_config, f)

model = YOLO('yolov8s.pt')

results = model.train(
    data='C:\\Users\\Yadeesh\\Downloads\\Tumor\\data.yaml',
    epochs=5,
    imgsz=416,
    batch=16,
    name='yolov8_brain_tumor_detection'
)

best_model_path = 'C:\\Users\\Yadeesh\\Downloads\\Tumor\\best.pt'
model = YOLO(best_model_path)

metrics = model.val()

print("\n\n--- Validation Metrics ---")
print(f"Mean Average Precision (mAP@50-95): {metrics.box.map:.4f}")
print(f"Mean Average Precision (mAP@50): {metrics.box.map50:.4f}")

metrics_data = {
    'Metric': ['mAP@50-95 (Overall)', 'mAP@50 (Overall)', 'Precision (glioma)', 'Recall (glioma)', 'mAP@50 (glioma)', 'mAP@50-95 (glioma)',
               'Precision (meningioma)', 'Recall (meningioma)', 'mAP@50 (meningioma)', 'mAP@50-95 (meningioma)',
               'Precision (pituitary)', 'Recall (pituitary)', 'mAP@50 (pituitary)', 'mAP@50-95 (pituitary)'],
    'Score': [metrics.box.map, metrics.box.map50,
              metrics.box.p[0], metrics.box.r[0], metrics.box.ap50[0], metrics.box.maps[0],
              metrics.box.p[1], metrics.box.r[1], metrics.box.ap50[1], metrics.box.maps[1],
              metrics.box.p[2], metrics.box.r[2], metrics.box.ap50[2], metrics.box.maps[2]]
}

metrics_df = pd.DataFrame(metrics_data)

metrics_df['Score'] = metrics_df['Score'].apply(lambda x: f"{x:.4f}")

display(metrics_df)


class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

all_image_paths_with_labels = []
for class_name in class_names:
    class_dir = os.path.join(source_dir, class_name)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths_with_labels.append((os.path.join(class_dir, filename), class_name))

num_images_to_display = 10
if len(all_image_paths_with_labels) > num_images_to_display:
    selected_images = random.sample(all_image_paths_with_labels, num_images_to_display)
else:
    selected_images = all_image_paths_with_labels

print(f"Displaying {len(selected_images)} random images with their labels...")


plt.figure(figsize=(15, 10))
for i, (image_path, class_name) in enumerate(selected_images):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        plt.subplot(2, 5, i + 1) 
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    else:
        print(f"Warning: Could not read image {image_path}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10)) 

for i, (image_path, class_name) in enumerate(selected_images):
    if os.path.exists(image_path):
        print(f"\nProcessing image: {image_path}")
        results = model(image_path)

        if results and results[0]:
            img_with_predictions = results[0].plot()

            plt.subplot(2, 5, i + 1)
            plt.imshow(img_with_predictions)
            plt.title(f"Prediction: {class_name}") 
            plt.axis('off')

        else:
            print(f"No detection results for image: {image_path}")
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(2, 5, i + 1)
                plt.imshow(img)
                plt.title(f"No prediction: {class_name}")
                plt.axis('off')
            else:
                 print(f"Warning: Could not read image {image_path}")

    else:
        print(f"Image not found: {image_path}")
        plt.subplot(2, 5, i + 1)
        plt.text(0.5, 0.5, "Image\nNot Found", horizontalalignment='center', verticalalignment='center', fontsize=10)
        plt.axis('off')


plt.tight_layout()
plt.show()

ground_truth_labels = []
predicted_labels = []
class_names = list(class_map.keys())

print("Evaluating image-level classification on the validation set...")
i=0
for img_path, true_class_idx in val_files:
    i+=1
    print(i)
    ground_truth_labels.append(true_class_idx)
    results = model(img_path, verbose=False)

    predicted_class_idx = class_map['notumor']

    if results and results[0].boxes:
        best_tumor_detection = None
        highest_confidence = -1

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_names[class_id] != 'notumor' and confidence > 0.5:
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_tumor_detection = class_id

        if best_tumor_detection is not None:
            predicted_class_idx = best_tumor_detection
        elif class_names[int(results[0].boxes[0].cls[0])] == 'notumor' and float(results[0].boxes[0].conf[0]) > 0.5:

             predicted_class_idx = class_map['notumor']


    predicted_labels.append(predicted_class_idx)

print("Evaluation complete.")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

accuracy = accuracy_score(ground_truth_labels, predicted_labels)
print(f"\nImage-level Classification Accuracy: {accuracy:.4f}")

print("\nImage-level Classification Report:")
print(classification_report(ground_truth_labels, predicted_labels, target_names=class_names))

conf_mat = confusion_matrix(ground_truth_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Image-level Classification Confusion Matrix')
plt.show()