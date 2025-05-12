# Tea Leaf Disease Detection - Project Overview

## 📊 Model Architecture

For this project, we implemented a Convolutional Neural Network (CNN) for the classification of tea leaf diseases. The chosen architecture is based on the **MobileNetV2** model, leveraging its efficiency and accuracy for image classification tasks.

- **Base Model**: MobileNetV2 (pre-trained on ImageNet, `include_top=False`)
- **Custom Head**:
  - GlobalAveragePooling2D
  - Dropout (rate: 0.5)
  - Dense layer with Softmax activation for classification into 7 disease categories
- **Optimization**:
  - Initial layers of MobileNetV2 are frozen during the first training phase
  - Fine-tuning is applied in a second training phase by unfreezing the base model layers

## 🛠 Preprocessing Steps

The following preprocessing techniques were applied to prepare the dataset:

1. **Image Resizing**: All images were resized to (224, 224, 3) to match the MobileNetV2 input shape.
2. **Rescaling**: Pixel values were normalized by scaling them to the [0, 1] range.
3. **Data Augmentation** (applied to training set):
   - Horizontal and vertical flipping
   - Rotation (up to 20 degrees)
   - Zoom and shift transformations

Data augmentation helps improve model generalization by increasing data diversity.

## 🔧 Techniques Used

- **Transfer Learning**:
  - The base MobileNetV2 model is initialized with pre-trained ImageNet weights.
  - Initially, only the custom classification head is trained, while the base model layers remain frozen.
  - Fine-tuning is then performed by progressively unfreezing selected layers to allow medium-level adjustments.
  - In the final phase, all layers are unfrozen and trained with a very low learning rate to refine the model on edge cases and improve generalization. (This phase has been dropped due to low improvements)


- **Callbacks**:
  - EarlyStopping: To prevent overfitting and stop training when validation loss no longer improves.
  - ReduceLROnPlateau: Dynamically reduce learning rate if validation performance stagnates.
  - ModelCheckpoint: Save the best model based on validation accuracy.

## 🔍 Observations & Challenges

### Observations
- Transfer learning significantly accelerated convergence and boosted accuracy on a limited dataset.
- Fine-tuning the entire network improved validation accuracy after initial plateauing.
- Data augmentation effectively mitigated overfitting tendencies observed in early epochs.

### Challenges
- The dataset is relatively small, increasing the risk of overfitting.
- Some disease classes exhibit visual similarities, making classification harder and prone to misclassification.
- Balancing between underfitting (too shallow training) and overfitting (too aggressive fine-tuning) required careful monitoring of validation metrics.