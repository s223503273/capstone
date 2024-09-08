To integrate the provided code into the GitHub documentation for your project, I'll organize the content into appropriate sections. This documentation will combine the YOLOv10 face mask detection details with the code you've shared for age detection using a ResNet50 model.

---

# Face Mask and Age Detection using YOLOv10 and ResNet50

This repository contains the code and instructions for training models to detect face masks and predict age groups using YOLOv10 and ResNet50.

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Training](#model-training)
5. [Inference](#inference)
6. [Scripts](#scripts)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Introduction

This project includes two tasks:
1. **Face Mask Detection**: Using YOLOv10 to detect whether people are wearing face masks correctly, incorrectly, or not at all.
2. **Age Group Classification**: Using ResNet50 to classify individuals into three age groups: Young, Middle, and Old.

## Environment Setup

To replicate this project, you'll need to install the required packages and set up the environment.

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/face-mask-age-detection.git
    cd face-mask-age-detection
    ```

2. Install dependencies:
    ```bash
    pip install -q git+https://github.com/THU-MIG/yolov10.git
    pip install -q supervision 
    pip uninstall -y wandb
    pip install -q --upgrade huggingface_hub
    pip install tensorflow opencv-python scikit-learn matplotlib seaborn pandas
    ```

## Dataset Preparation

### 1. Face Mask Detection Dataset

Prepare the dataset for face mask detection:

- **Create the necessary folder structure**:
    ```python
    import os
    
    def create_folder_structure(base_path):
        folders = [
            'data/train/images', 'data/train/labels',
            'data/val/images', 'data/val/labels',
            'data/test/images', 'data/test/labels'
        ]
        for folder in folders:
            os.makedirs(os.path.join(base_path, folder), exist_ok=True)

    create_folder_structure(os.getcwd())
    ```

- **Convert XML annotations to YOLO format**:
    ```python
    import xml.etree.ElementTree as ET
    
    def convert_bbox(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        return (x * dw, y * dh, w * dw, h * dh)

    def convert_annotation(xml_path, output_path, classes):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        with open(output_path, 'w') as out_file:
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                bb = convert_bbox((w, h), b)
                out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")
    ```

- **Process and split the dataset**:
    ```python
    from sklearn.model_selection import train_test_split
    import shutil
    
    def process_dataset(dataset_path, output_path):
        classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        image_folder = os.path.join(dataset_path, 'images')
        annotation_folder = os.path.join(dataset_path, 'annotations')
        
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        
        # Split the dataset
        train_val, test = train_test_split(image_files, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.2, random_state=42)
        
        # Process and move files
        for split, files in [('train', train), ('val', val), ('test', test)]:
            for file in files:
                # Image
                src_img = os.path.join(image_folder, file)
                dst_img = os.path.join(output_path, f'data/{split}/images', file)
                shutil.copy(src_img, dst_img)
                
                # Annotation
                xml_file = os.path.splitext(file)[0] + '.xml'
                src_xml = os.path.join(annotation_folder, xml_file)
                dst_txt = os.path.join(output_path, f'data/{split}/labels', os.path.splitext(file)[0] + '.txt')
                convert_annotation(src_xml, dst_txt, classes)
        
        # Create data.yaml
        yaml_content = {
            'train': f'{output_path}/data/train/images',
            'val': f'{output_path}/data/val/images',
            'test': f'{output_path}/data/test/images',
            'nc': len(classes),
            'names': classes
        }
        
        with open(os.path.join(output_path, 'data', 'data.yaml'), 'w') as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False)

    process_dataset('/kaggle/input/face-mask-detection', os.getcwd())
    ```

### 2. Age Detection Dataset

Prepare the dataset for age detection:

- **Load the dataset**:
    ```python
    df = pd.read_csv(r"/kaggle/input/faces-age-detection-dataset/faces/train.csv")
    ```

- **Map age groups**:
    ```python
    age_group = {
        "YOUNG": 0,
        "MIDDLE": 1,
        "OLD": 2
    }

    df['target'] = df['Class'].map(age_group)
    ```

- **Visualize class distribution**:
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    class_count = df['Class'].value_counts()
    class_count.plot.pie(autopct="%.2f", colors=sns.color_palette("Blues_r"))
    plt.show()
    ```

- **Load and display sample images**:
    ```python
    import os
    import cv2

    folder_path = '/kaggle/input/faces-age-detection-dataset/faces/Train'
    files = os.listdir(folder_path)

    N = 5 
    plt.figure(figsize=(20, 10))

    for i, file in enumerate(files[N+5:N + 10]):
        image_path = os.path.join(folder_path, file)
        img = plt.imread(image_path)
        plt.subplot(1, N, i+1)
        plt.imshow(img)
        plt.title(df.loc[df['ID'] == file, "Class"].item())
        plt.xlabel(img.shape)
        plt.axis('off')

    plt.show()
    ```

- **Augment and preprocess images**:
    ```python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import numpy as np
    import random
    import cv2

    imggen = ImageDataGenerator(
        rescale=1./255,
        brightness_range=(0.4, 0.55),
        horizontal_flip=True,
        width_shift_range=0.22,
    )

    def augment_image(image, target_size):
        # Rotation
        angle = random.uniform(-30, 30)
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Flipping
        if random.choice([True, False]):
            image = cv2.flip(image, 1)  # Horizontal flip
        if random.choice([True, False]):
            image = cv2.flip(image, 0)  # Vertical flip
        
        # Scaling
        scale_factor = random.uniform(0.8, 1.2)
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        
        # Ensure image size is consistent
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Translation
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        M = np.float32([[1, 0, x], [0, 1, y]])
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Brightness adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], random.uniform(0.5, 1.

5))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image

    plt.figure(figsize=(15, 5))
    for i in range(10):
        img = plt.imread(image_path)
        augmented_img = augment_image(img, target_size=(224, 224))
        plt.subplot(2, 5, i+1)
        plt.imshow(augmented_img)
        plt.axis('off')
    plt.show()
    ```

## Model Training

### 1. Face Mask Detection (YOLOv10)

Train the YOLOv10 model for face mask detection:

```bash
yolo task=detect mode=train epochs=50 batch=32 plots=True \
model=$(pwd)/weights/yolov10n.pt \
data=$(pwd)/data/data.yaml
```

### 2. Age Detection (ResNet50)

Train the ResNet50 model for age detection:

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential

def build_model(input_shape=(224, 224, 3), num_classes=3):
    base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)
```

## Inference

### 1. Face Mask Detection

Run inference on a new image:

```python
yolo task=detect mode=predict model=weights/yolov10n.pt source=input_image.jpg
```

### 2. Age Detection

Run inference using the trained ResNet50 model:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_age(model, img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    age_group = np.argmax(prediction)
    
    return age_group

predicted_age_group = predict_age(model, 'path_to_image.jpg')
print("Predicted Age Group:", predicted_age_group)
```

## Scripts

This repository includes several scripts to help with dataset preparation, training, and inference. Each script is described in detail in the respective sections above.

## Results

### 1. Face Mask Detection

The model's performance metrics such as Precision, Recall, and mAP50 are logged during training.

### 2. Age Detection

Plot the accuracy and loss curves:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
```

## Conclusion

Both models—YOLOv10 for face mask detection and ResNet50 for age classification—performed well on their respective tasks, demonstrating the effectiveness of modern deep learning techniques in image classification and object detection.

---

This README file now incorporates all the code you provided, structured for clarity and ease of use for anyone replicating your work.
