🚗 Car Object Detection using CNN (ResNet50)

This project implements a Car Object Detection System using Deep Learning with a CNN based on ResNet50.
It detects whether a car is present in an image and predicts its bounding box.

📌 Overview

The model is designed as a multi-output neural network:

Classification Task → Detects presence of a car
Localization Task → Predicts bounding box coordinates

This project also compares multiple optimizers to evaluate performance.

📂 Project Structure
car_object_detection/
│
├── data/
│   ├── training_images/
│   ├── testing_images/
│   ├── negatives/
│   └── train_solution_bounding_boxes.csv
│
├── models/                # Saved models (ignored in Git)
│
├── src/
│   ├── data_loader.py     # Data loading & preprocessing
│   ├── model.py           # Model architecture (ResNet50)
│   ├── train.py           # Training pipeline
│   └── plot_graph.py      # Accuracy comparison graph
│
├── app/
│   └── app.py             # Streamlit web application
│
├── results/               # Output results & graphs
│
├── main.py                # Entry point
├── requirements.txt
└── README.md
⚙️ Features
Car detection using CNN (ResNet50)
Bounding box prediction (object localization)
Multi-output model (classification + regression)
Negative sample handling to reduce false positives
Optimizer comparison (Adam, SGD, RMSprop)
Interactive UI using Streamlit
📦 Installation
git clone https://github.com/ABHISHEKKHOPADE/Car-Object-Detection.git
cd Car-Object-Detection

pip install -r requirements.txt
▶️ How to Run
1️⃣ Train the Model
python main.py

This will:

Train models using Adam, SGD, and RMSprop
Save trained models in the models/ folder
2️⃣ Run the Web App
streamlit run app/app.py

Then:

Upload an image
Select a trained model
View detection results with bounding box
3️⃣ Generate Accuracy Comparison Graph
python src/plot_graph.py

Output:

results/accuracy_comparison.png
📊 Output
Bounding box drawn on detected car
Confidence score displayed
Accuracy comparison graph
🧠 Model Details
Backbone: ResNet50 (Transfer Learning)
Input Size: 224 × 224

Loss Functions:

Classification → Binary Crossentropy
Bounding Box → Huber Loss
🎯 Results
Accurate car detection
Reduced false positives using negative samples
Adam optimizer performed best
📥 Model Weights

Due to GitHub file size limits, trained models are not included.

👉 Download models from:
(Add your Google Drive link here)

🚀 Future Improvements
Multi-object detection (YOLO / SSD)
Real-time detection using OpenCV
Larger and more diverse dataset
Model optimization for deployment
👨‍💻 Author

Abhishek Khopade

⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

🔒 .gitignore (Recommended)
venv/
venv311/
models/
*.h5
__pycache__/
*.pyc
.ipynb_checkpoints/
