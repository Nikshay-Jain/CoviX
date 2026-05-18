# CoviX
**COVID-19 Detection from Chest X-rays using VGG-16 Deep Learning**

## Features
- **VGG-16 CNN Architecture**: Transfer learning on pre-trained VGG-16 model for COVID-19 classification
- **Binary Classification**: Distinguishes between COVID-positive and COVID-negative chest X-rays
- **High Accuracy**: Achieves robust performance with data augmentation and optimized training
- **Model Persistence**: Trained model saved as `model1.h5` for inference and deployment

## Quick Start/Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nikshay-Jain/CoviX.git
   cd CoviX
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow keras numpy pandas matplotlib scikit-learn
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook covid-19-detection-using-vgg-16.ipynb
   ```

## Usage
Open the notebook and execute cells sequentially. The model trains on the `dataset/` directory (organized as `covid/` and `not_covid/` folders). For inference on new X-rays, load the saved model:

```python
from tensorflow.keras.models import load_model
model = load_model('model1.h5')
# Predict on preprocessed image
prediction = model.predict(preprocessed_image)
```
