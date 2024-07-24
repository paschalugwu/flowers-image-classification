# Image Classifier for Flower Species

This README provides a comprehensive guide to developing an AI application for classifying different species of flowers using deep learning. The project involves training a classifier to recognize flower species, which could be utilized in applications such as a mobile app that identifies flowers from camera images.

## Project Overview

In this project, you will:
1. **Load and preprocess the image dataset**
2. **Train an image classifier**
3. **Use the trained classifier to predict image content**

By the end of the project, you will have a command-line application capable of training on any set of labeled images. The skills and methods developed here can be applied to various other image classification tasks.

## Requirements

- Python (3.x)
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- Matplotlib
- JSON

## Steps

### 1. Importing Packages

Start by importing the necessary packages. Keeping all imports at the beginning of your code is a good practice.

```python
# TODO: Import necessary packages
```

### 2. Load and Preprocess the Dataset

**Dataset Overview:**
- The dataset consists of images split into three sets: training, validation, and testing.
- The training set will be augmented with transformations such as random scaling, cropping, and flipping.
- The validation and testing sets will be resized to 224x224 pixels without transformations.

**Tasks:**
- Define transformations for each dataset.
- Load the datasets using `ImageFolder`.
- Create data loaders.

```python
# TODO: Define data transformations
# TODO: Load datasets with ImageFolder
# TODO: Define data loaders
```

### 3. Label Mapping

Load a JSON file containing the mapping from category labels to category names. This mapping will help in interpreting the classifier's output.

```python
import json

# Load label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

### 4. Build and Train the Classifier

**Tasks:**
- Use a pre-trained network (e.g., VGG) as the feature extractor.
- Define and train a new feed-forward network as the classifier.
- Tune hyperparameters and track performance on the validation set.

**Note:** Ensure only the classifier layers are trained while the pre-trained network weights remain frozen.

```python
# TODO: Build and train the classifier
```

### 5. Test the Network

Evaluate the trained network on the test dataset to estimate its performance on unseen data. Aim for an accuracy around 70%.

```python
# TODO: Evaluate the network on the test set
```

### 6. Save the Checkpoint

Save the trained model, including the classifier weights and additional information such as class-to-index mappings. This allows for easy future use and inference.

```python
# TODO: Save the model checkpoint
```

### 7. Load the Checkpoint

Write a function to load the model checkpoint and reconstruct the model for future use.

```python
# TODO: Implement checkpoint loading function
```

### 8. Image Preprocessing for Inference

**Tasks:**
- Write a function to preprocess images for the model: resizing, cropping, normalizing.
- Convert images from PIL format to PyTorch tensors.

```python
def process_image(image):
    ''' Preprocess a PIL image for a PyTorch model. '''
    # TODO: Implement preprocessing steps
```

### 9. Class Prediction

Implement a function to use the trained model for making predictions. This function should return the top K probable classes and their associated probabilities.

```python
def predict(image_path, model, topk=5):
    ''' Predict the class of an image using a trained model. '''
    # TODO: Implement prediction function
```

### 10. Sanity Checking

Verify the model's predictions by displaying the input image along with the top 5 predicted classes and their probabilities. Use Matplotlib to visualize these results.

```python
# TODO: Display an image along with the top 5 predicted classes
```

## Notes

- Ensure that the workspace remains active during long-running tasks to prevent disconnection.
- If the model checkpoint exceeds 1 GB, consider reducing the size of the hidden layers to avoid saving issues.

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision Documentation](https://pytorch.org/vision/stable/index.html)
- [PIL Documentation](https://pillow.readthedocs.io/en/stable/)
