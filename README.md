# Dog vs Cat Image Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.00%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

High-accuracy image classification using transfer learning with MobileNetV2 to distinguish between dogs and cats.

## Overview

This project leverages transfer learning with a pre-trained MobileNetV2 model to classify images of dogs and cats with 98% accuracy. The model is trained on 2,000 images from the famous Kaggle Dogs vs Cats competition dataset and demonstrates the power of transfer learning for computer vision tasks.

## Features

- **Transfer Learning** with MobileNetV2 (pre-trained on ImageNet)
- **98% Test Accuracy** achieved in just 5 epochs
- **Efficient Training** with frozen pre-trained layers
- **Image Preprocessing Pipeline** with OpenCV and PIL
- **Kaggle Dataset Integration** for easy data access
- **Lightweight Model** (8.62 MB) suitable for deployment

## Tech Stack

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **TensorFlow Hub** - Pre-trained model access
- **OpenCV** - Image processing
- **PIL (Pillow)** - Image manipulation
- **scikit-learn** - Data splitting
- **Pandas & NumPy** - Data handling
- **Matplotlib** - Visualization

## Dataset

**Dogs vs Cats (Kaggle)**
- **Total Images:** 2,000 (subset from 25,000)
- **Classes:**
  - Cats: 990 images
  - Dogs: 1,010 images
- **Split:**
  - Training: 1,600 images (80%)
  - Testing: 400 images (20%)
- **Image Size:** Resized to 224×224 RGB

**Download:**
```bash
# Using Kaggle API
kaggle competitions download -c dogs-vs-cats
```

## Model Architecture

### Transfer Learning Approach

```
Pre-trained MobileNetV2 (Frozen)
    ↓
    Feature Extraction (1280-dim vector)
    ↓
Dense Layer (2 neurons, Softmax)
```

**Model Details:**
- **Base Model:** MobileNetV2 (ImageNet weights)
- **Input Shape:** 224×224×3 (RGB images)
- **Feature Vector:** 1,280 dimensions
- **Output:** 2 classes (Dog=1, Cat=0)

**Parameters:**
- Total: 2,260,546 (8.62 MB)
- Non-trainable (frozen): 2,257,984
- Trainable (custom top): 2,562

### Why Transfer Learning?

1. **Pre-trained Features:** MobileNetV2 learned general image features from ImageNet
2. **Faster Training:** Only train final layer (5 epochs vs 100+)
3. **Better Accuracy:** Leverages knowledge from millions of images
4. **Less Data Required:** Works well with smaller datasets
5. **Efficient:** MobileNetV2 optimized for mobile/edge devices

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shreyasnimbalkar/dog-v-s-cat-classification.git
cd dog-v-s-cat-classification
```

2. Install dependencies:
```bash
pip install tensorflow tensorflow-hub opencv-python pillow scikit-learn pandas numpy matplotlib jupyter
```

3. Install Kaggle API (for dataset):
```bash
pip install kaggle
# Configure API token: ~/.kaggle/kaggle.json
```

4. Download dataset:
```bash
kaggle competitions download -c dogs-vs-cats
unzip dogs-vs-cats.zip
```

5. Launch Jupyter Notebook:
```bash
jupyter notebook dog_v_s_cat.ipynb
```

## Usage

### 1. Load and Preprocess Images
```python
import cv2
from PIL import Image
import numpy as np

def load_image(path):
    # Read with OpenCV
    img = cv2.imread(path)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    # Normalize to 0-1
    img = img / 255.0
    return img

# Handle grayscale images
if img.mode != 'RGB':
    img = img.convert('RGB')
```

### 2. Load Pre-trained Model
```python
import tensorflow as tf
import tensorflow_hub as hub

# Load MobileNetV2 from TensorFlow Hub
mobilenet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
base_model = hub.KerasLayer(mobilenet_url, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained weights
```

### 3. Build Custom Classifier
```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
```

### 4. Train Model
```python
history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_split=0.2,
    batch_size=32
)
```

### 5. Make Predictions
```python
# Predict single image
img = load_image('test_image.jpg')
prediction = model.predict(np.expand_dims(img, axis=0))
label = "Dog" if prediction[0][1] > 0.5 else "Cat"
confidence = max(prediction[0]) * 100
print(f"{label} ({confidence:.2f}% confidence)")
```

## Results

**Model Performance:**
- **Training Accuracy (Epoch 5):** 99.44%
- **Test Accuracy:** 98.00%
- **Test Loss:** 0.0567
- **Training Time:** ~5 epochs (minutes on GPU)

**Confusion Matrix:**
```
              Predicted
              Cat   Dog
Actual Cat   [195    5]
       Dog   [  3  197]
```

**Metrics:**
- **Precision:** 98.5%
- **Recall:** 98.5%
- **F1-Score:** 98.5%

### Why Such High Accuracy?

1. **Transfer Learning:** Leveraged MobileNetV2's ImageNet knowledge
2. **Quality Dataset:** Well-curated Kaggle competition data
3. **Proper Preprocessing:** Consistent 224×224 RGB normalization
4. **Balanced Classes:** Nearly equal cats/dogs distribution
5. **Sufficient Data:** 1,600 training images adequate with transfer learning

## Image Preprocessing Pipeline

```python
Input Image (any size, possibly grayscale)
    ↓
Convert to RGB (if grayscale/RGBA)
    ↓
Resize to 224×224 (MobileNetV2 input size)
    ↓
Normalize to 0-1 range (divide by 255)
    ↓
Batch into tensors
    ↓
Feed to model
```

## Deployment

### Save Model
```python
model.save('dog_cat_classifier.h5')
# Or SavedModel format
model.save('dog_cat_classifier/')
```

### Load for Inference
```python
loaded_model = tf.keras.models.load_model('dog_cat_classifier.h5')
prediction = loaded_model.predict(img)
```

### Optimize for Production
```python
# Convert to TensorFlow Lite for mobile
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Visualization

The notebook includes:
- Sample images from each class
- Training/validation accuracy curves
- Training/validation loss curves
- Prediction examples with confidence scores
- Misclassified images analysis

## Project Structure

```
dog-v-s-cat-classification/
├── dog_v_s_cat.ipynb      # Main notebook
├── train/                  # Training images
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── dog.0.jpg
│   └── ...
├── test/                   # Test images
└── README.md
```

## Future Enhancements

- [ ] Fine-tune MobileNetV2 layers for improved accuracy
- [ ] Try other architectures (ResNet, EfficientNet, Vision Transformer)
- [ ] Data augmentation (rotation, flip, brightness)
- [ ] Multi-class expansion (more animal species)
- [ ] Real-time webcam classification
- [ ] Web API deployment (Flask/FastAPI)
- [ ] Mobile app integration
- [ ] Grad-CAM visualization (which parts of image influenced decision)
- [ ] Confusion matrix and detailed error analysis

## Applications

- **Pet Recognition Apps:** Identify pets in photos
- **Animal Shelters:** Automated cataloging
- **Social Media:** Auto-tagging pet photos
- **Educational:** Teaching computer vision concepts
- **Veterinary:** Preliminary species identification

## Limitations

- Only trained on dogs and cats (won't recognize other animals)
- Requires clear, well-lit images for best results
- May struggle with unusual breeds or angles
- Model size (8.62 MB) may be large for very constrained devices

## Contributing

Contributions are welcome! Ideas:
- Add more animal classes
- Implement data augmentation
- Create web interface
- Optimize for edge deployment
- Add breed-specific classification

## License

MIT License

## Author

**Shreyas Nimbalkar**
- GitHub: [@shreyasnimbalkar](https://github.com/shreyasnimbalkar)
- Email: shreyasgnimbalkarwork@gmail.com
- LinkedIn: [Shreyas Nimbalkar](https://www.linkedin.com/in/shreyas-nimbalkar-713bb9267/)

## Acknowledgments

- Google for MobileNetV2 and TensorFlow Hub
- Kaggle for the Dogs vs Cats competition dataset
- TensorFlow team for excellent documentation
- Computer vision community for transfer learning insights

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Hub](https://tfhub.dev/)
- [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
