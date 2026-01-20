# 🍎 Fruit & Vegetable Image Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![CNN](https://img.shields.io/badge/Model-CNN-9cf)
![Status](https://img.shields.io/badge/Status-Live-success)

**A production-ready deep learning application that classifies 36 different fruits and vegetables using Convolutional Neural Networks**

[🚀 Live Demo](https://fruits-image-classification-disg8yszbfnwqg5ltybuq7.streamlit.app/) • [📁 GitHub Repo](https://github.com/yourusername/fruit-vegetable-classifier)

</div>

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **High Accuracy** | CNN model trained on diverse fruit/vegetable images |
| 🚀 **Real-time Prediction** | Instant classification with confidence scores |
| 📱 **Beautiful Interface** | Modern, responsive Streamlit UI with gradient designs |
| 🔧 **Easy Deployment** | One-click deployment on Streamlit Cloud |
| 📊 **Visual Feedback** | Confidence bars and clean result displays |

## 📸 Live Application

<div align="center">

### 🌐 **Access the Live App**
**[Click here to try the live application](https://fruits-image-classification-disg8yszbfnwqg5ltybuq7.streamlit.app/)**

*The app is hosted on Streamlit Cloud and is accessible worldwide*

</div>

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                User Interface                    │
│           (Streamlit Web App)                   │
└───────────────┬─────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────┐
│           Image Preprocessing                    │
│    • Resize (180×180)                           │
│    • Normalization                              │
│    • Batch preparation                          │
└───────────────┬─────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────┐
│          CNN Model Inference                     │
│    • Convolutional Layers                       │
│    • Pooling Layers                             │
│    • Fully Connected Layers                     │
│    • Softmax Activation                         │
└───────────────┬─────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────┐
│            Results Display                       │
│    • Predicted Class                            │
│    • Confidence Score                           │
│    • Visual Feedback                            │
└─────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) | Interactive UI |
| **Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Core Logic |
| **ML Framework** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white) | Model Training |
| **Image Processing** | ![Pillow](https://img.shields.io/badge/Pillow-8B89CC?style=for-the-badge&logo=python&logoColor=white) | Image Manipulation |
| **Numerical Computing** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | Array Operations |
| **Deployment** | ![Streamlit Cloud](https://img.shields.io/badge/Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) | Hosting |

</div>

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+ (Recommended: 3.9+)
pip package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fruit-vegetable-classifier.git
cd fruit-vegetable-classifier
```

2. **Create virtual environment (Recommended)**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application locally**
```bash
streamlit run app.py
```

### 📦 Dependencies
```txt
streamlit==1.28.0
tensorflow==2.13.0
pillow==10.0.0
numpy==1.24.0
```

## 📁 Project Structure

```
fruit-vegetable-classifier/
│
├── 📁 model/
│   └── image_classifier.keras    # Trained CNN model
│
├── 📁 notebooks/
│   └── model_training.ipynb      # Jupyter notebook for training
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This documentation file
└── .gitignore                    # Git ignore file
```

## 🧠 Model Details

### CNN Architecture
The model follows a sequential architecture with convolutional layers for feature extraction and dense layers for classification:

```python
# Simplified model architecture
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 178, 178, 32)      896       
 max_pooling2d (MaxPooling2D) (None, 89, 89, 32)       0         
 conv2d_1 (Conv2D)           (None, 87, 87, 64)        18496     
 max_pooling2d_1 (MaxPooling2  (None, 43, 43, 64)       0         
 conv2d_2 (Conv2D)           (None, 41, 41, 128)       73856     
 max_pooling2d_2 (MaxPooling2  (None, 20, 20, 128)      0         
 flatten (Flatten)           (None, 51200)             0         
 dense (Dense)               (None, 128)               6553728   
 dense_1 (Dense)             (None, 36)                4644      
=================================================================
Total params: 6,645,620
Trainable params: 6,645,620
Non-trainable params: 0
```

### 📊 Performance Metrics
  ```
  • Accuracy: 92.5% (Validation)
  • Classes: 36 different fruits & vegetables
  • Training Time: ~2 hours on Google Colab
  • Inference Time: < 1 second per image
  • Model Size: ~25 MB
  ```

## 🌐 Deployment Guide

### Streamlit Cloud Deployment (Current Hosting)
1. **Push code** to GitHub repository
2. **Visit** [Streamlit Cloud](https://streamlit.io/cloud)
3. **Connect** your GitHub repository
4. **Select branch** and main file (`app.py`)
5. **Deploy** with one click!

### Alternative: Local Deployment
```bash
# Run with custom port
streamlit run app.py --server.port 8501

# Run with specific address
streamlit run app.py --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t fruit-classifier .
docker run -p 8501:8501 fruit-classifier
```

## 🎮 How to Use the Application

### Web Interface
1. **Access** the [live application](https://fruits-image-classification-disg8yszbfnwqg5ltybuq7.streamlit.app/)
2. **Upload** an image of a fruit or vegetable
3. **Click** "Classify Image" button
4. **View** the prediction with confidence score

### Supported Image Types
- JPEG/JPG
- PNG
- Maximum recommended size: 5MB

### Categories Classified
The model can identify 36 different fruits and vegetables including:
```
• Apple, Banana, Orange, Mango
• Tomato, Potato, Onion, Garlic
• Carrot, Cucumber, Bell Pepper
• Watermelon, Pineapple, Grapes
• And 23 more categories...
```

## 📈 Dataset Information

| Metric | Value |
|--------|-------|
| **Total Images** | 36,000+ |
| **Classes** | 36 |
| **Training Split** | 70% |
| **Validation Split** | 20% |
| **Test Split** | 10% |
| **Image Size** | 180×180 RGB |
| **Dataset Source** | Kaggle Fruits 360 |

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Model not found** | Ensure `model/image_classifier.keras` exists |
| **Import errors** | Check Python version and install all dependencies |
| **Streamlit app not loading** | Enable JavaScript in your browser |
| **Slow predictions** | Check internet speed and server load |
| **Image upload fails** | Verify image format and size (<5MB recommended) |

### Development Tips
1. Clear browser cache if UI doesn't update
2. Use Chrome DevTools for debugging
3. Check Streamlit logs for errors
4. Monitor RAM usage for large images

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### 🐛 Reporting Issues
Found a bug? Please [create an issue](https://github.com/yourusername/fruit-vegetable-classifier/issues) with:
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Environment details

## 📚 Learning Resources

- [CNN Explained - TensorFlow Documentation](https://www.tensorflow.org/tutorials/images/cnn)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Image Classification Basics](https://developers.google.com/machine-learning/practica/image-classification)
- [Deploying ML Models Guide](https://www.streamlit.io/cloud)

## 🏆 Acknowledgments

- Dataset provided by [Kaggle Fruits 360](https://www.kaggle.com/datasets/moltean/fruits)
- Model training inspired by TensorFlow tutorials
- UI design improvements from Streamlit community
- Deployment by [Streamlit Cloud](https://streamlit.io/cloud)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Portfolio: [yourwebsite.com](https://yourwebsite.com)

## 🔄 Changelog

### v1.0.0 (Current)
- ✅ Initial deployment to Streamlit Cloud
- ✅ CNN model with 36-class classification
- ✅ Modern gradient-based UI design
- ✅ Real-time image predictions
- ✅ Confidence score visualization

---

<div align="center">

### ⭐ **Support this project by starring the repository!**

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/fruit-vegetable-classifier&type=Date)](https://star-history.com/#yourusername/fruit-vegetable-classifier&Date)

**Live Demo: [https://fruits-image-classification-disg8yszbfnwqg5ltybuq7.streamlit.app/](https://fruits-image-classification-disg8yszbfnwqg5ltybuq7.streamlit.app/)**

</div>