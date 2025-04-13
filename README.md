# Fire-Detection-
Fire Detection in Images
This project explores image classification techniques to detect fire in images using deep learning. It is designed as an educational and practical approach for applying computer vision to real-world emergency response problems.

Overview
Wildfires and accidental fires cause devastating damage to lives, property, and nature. Automatic fire detection using image data can significantly improve response times. In this notebook, we build a Convolutional Neural Network (CNN) model to classify images as either containing fire or not.

Dataset
The dataset used consists of two main classes:
Fire: Images that clearly show fire or flames.
No_Fire: Images without fire.
These images have been preprocessed and split into training, validation, and test sets.

Project Structure
Data Loading & Preprocessing
Load images, resize them, normalize, and apply augmentation for training.
Model Building
A CNN architecture is implemented using TensorFlow/Keras. It includes convolutional, pooling, and dense layers optimized for binary classification.
Training & Validation
The model is trained using the training dataset and validated on the validation dataset. Accuracy and loss curves are plotted.
Evaluation
Model performance is evaluated on the test set using classification metrics like accuracy, precision, recall, and F1-score.
Prediction & Visualization
Predictions are visualized for sample test images to demonstrate the model’s effectiveness.

Tech Stack
Python
TensorFlow / Keras
OpenCV
Matplotlib / Seaborn
NumPy / Pandas

Results
The model achieved promising accuracy on the test set, showing it can reliably detect fire in images. Performance may be further improved using techniques like transfer learning, ensemble models, or using larger datasets.

Future Work
Integrate with real-time video stream (e.g., CCTV)
Deploy the model using a web interface or mobile app
Use object detection models like YOLO or Faster R-CNN for fire localization

License
This project is for educational and research purposes.

