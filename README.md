🏥 Multimodal Disease Prediction System
A machine learning-based system that predicts diseases using two different input modalities:
📝 Symptom-based Text Input (NLP + ML)
🖼 Image-based Input (Computer Vision + Transfer Learning)
The system integrates both approaches into a unified prediction framework with a user-friendly GUI.
🚀 Features
Multimodal prediction (Text + Image)
TF-IDF based symptom analysis
Transfer Learning using ResNet50
Custom-built GUI for easy interaction
Real-time inference
High accuracy models
🧠 Model Architecture
1️⃣ Text-Based Disease Prediction
Text preprocessing (cleaning, tokenization)
TF-IDF vectorization
Machine Learning classifiers:
Logistic Regression
Random Forest
SVM (final selected model)
Achieved 97% accuracy
2️⃣ Image-Based Disease Prediction
Transfer Learning using ResNet50
Image preprocessing & augmentation
Fine-tuned final classification layers
Achieved 85% accuracy
🖥 GUI Interface
A custom graphical user interface was developed (using Python GUI framework such as Tkinter/PyQt) to:
Enter symptoms manually
Upload medical images
Display prediction results instantly
Provide probability/confidence scores
🛠 Tech Stack
Python
Scikit-learn
TensorFlow / Keras
OpenCV
ResNet50 (Transfer Learning)
GUI Framework (Tkinter / PyQt)
NumPy
Pandas
📊 Results
Modality	Accuracy
Text-Based Model	97%
Image-Based Model	85%
