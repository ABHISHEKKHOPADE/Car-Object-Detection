🚗 Car Object Detection using CNN (ResNet50)

This project implements a Car Object Detection System using Deep Learning (CNN) with ResNet50.
It detects whether a car is present in an image and predicts its bounding box.

📂 Project Structure
car_object_detection/
│
├── data/
│   ├── training_images/
│   ├── testing_images/
│   ├── negatives/
│   └── train_solution_bounding_boxes.csv
│
├── models/
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── plot_graph.py
│
├── app/
│   └── app.py
│
├── results/
│
├── main.py
└── requirements.txt
⚙️ Features
Car detection using CNN
Bounding box prediction
Multi-output model (classification + localization)
Negative dataset handling
Optimizer comparison (Adam, SGD, RMSprop)
Streamlit UI for testing
📦 Installation
pip install -r requirements.txt
▶️ How to Run
1️⃣ Train the Model
python main.py

👉 This will:

Train models (Adam, SGD, RMSprop)
Save them in /models
2️⃣ Run the UI
streamlit run app/app.py

👉 Then:

Upload an image
Select model
View detection results
3️⃣ Generate Accuracy Graph
python src/plot_graph.py

👉 Output saved in:

results/accuracy_comparison.png
📊 Output
Bounding box drawn on detected car
Confidence score displayed
Accuracy comparison graph
🧠 Model Details
Backbone: ResNet50
Loss:
Binary Crossentropy (classification)
Huber Loss (bounding box)
Input size: 224×224
🎯 Results
Accurate car detection
Reduced false positives using negative samples
Adam optimizer performs best
🚀 Future Improvements
Multi-object detection (YOLO)
Real-time detection
Better dataset
👨‍💻 Author

Abhishek Khopade