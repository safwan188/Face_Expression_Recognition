# Facial Expression Recognition Project

## Overview
This project focuses on developing a robust system for facial expression recognition using deep learning techniques. The system includes data preprocessing, data augmentation, model training, evaluation, and a graphical user interface for real-time facial expression detection.

## Project Structure

### Data Preprocessing and Visualization
- **Counting Images:** Count the number of images per expression in the training and testing datasets to understand class distribution.
- **Visualizing Distribution:** Create visualizations to display the distribution of images across different expressions, highlighting any class imbalances.

### Data Augmentation
- **Balancing Classes:** Apply data augmentation techniques to balance the dataset by creating additional samples for underrepresented classes.
- **Augmentation Strategies:** Augment the dataset using two strategies:
  1. **Max Augmentation:** Increase the number of images in each class to match the class with the highest number of samples.
  2. **Median Augmentation:** Increase the number of images in each class to approximate the median number of samples across all classes.

### Model Training
- **CNN Architecture:** Develop a convolutional neural network (CNN) model tailored for facial expression recognition.
- **Training Process:** Train the model on both the original and augmented datasets to improve performance and generalizability.
- **Model Versions:** Train two distinct models using the augmented datasets:
  1. **Max Augmented Dataset Model:** Trained on the maximally augmented dataset.
  2. **Median Augmented Dataset Model:** Trained on the median-augmented dataset.
- **Model Saving:** Save the trained models' weights to files for future use:
  - `max_conv_model1.h5`
  - `median_conv_model1.h5`

### Model Evaluation
- **Performance Metrics:** Evaluate the model performance using classification reports and confusion matrices.
- **Visualization:** Visualize metrics such as precision, recall, and F1-score for each expression to assess the models' effectiveness.

### Real-Time Facial Expression Detection
- **GUI Application:** Implement a graphical user interface (GUI) using Tkinter for real-time facial expression detection via webcam.
- **Face Detection:** Use OpenCV’s pre-trained Haar cascades for face detection, integrated with the trained CNN models for expression classification.

## Dependencies
- Python 3.x
- TensorFlow/Keras
- OpenCV
- Matplotlib
- Seaborn
- Numpy
- Tkinter
- PIL (Pillow)

## Files in the Project
- `face_expression_gui.py`: The script for the real-time facial expression detection GUI application.
- `jj.ipynb`: The Jupyter notebook containing the code for data preprocessing, augmentation, model training, and evaluation.
- `max_conv_model1.h5`: Saved weights of the model trained on the maximally augmented dataset.
- `median_conv_model1.h5`: Saved weights of the model trained on the median-augmented dataset.
- `readme.md`: This README file containing project documentation.

## Installation and Usage

### Prerequisites
Ensure that you have the following dependencies installed:
```sh
pip install tensorflow keras opencv-python matplotlib seaborn numpy pillow
```

### Running the Project

1. **Data Preprocessing and Augmentation:**
   - Use the Jupyter notebook `jj.ipynb` to preprocess the data and perform augmentation.
  
2. **Model Training:**
   - Train the models using the provided code in the notebook and save the weights to `max_conv_model1.h5` and `median_conv_model1.h5`.

3. **Model Evaluation:**
   - Evaluate the trained models using the evaluation code in the notebook.

4. **Real-Time Detection:**
   - Run `face_expression_gui.py` to start the GUI application for real-time facial expression detection:
   ```sh
   python face_expression_gui.py
   ```

## Acknowledgments
This project is inspired by the article "Facial Expression Recognition Using KERAS" and utilizes the FER-2013 dataset available on Kaggle. Special thanks to the contributors of these resources for their valuable work.
note:trainging and test data isnt available in the project you can find it on the web or ask for it .
