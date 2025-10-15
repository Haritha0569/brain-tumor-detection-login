# 🧠 Brain Tumor Detection using CNN - Cleaned Workspace

A mobile-optimized AI application for brain tumor detection using Convolutional Neural Networks (CNN) with Gradio interface.

## ✨ Essential Files (Cleaned Workspace)

After cleanup, your workspace contains only the working files:

```
Brain Tumor Detection using CNN/
├── brain_tumor_app.py              # 📱 Main mobile app (WORKING)
├── improved_model_training.py      # 🧠 Model training script (WORKING)  
├── improved_brain_tumor_model.h5   # 🤖 Trained CNN model (WORKING)
├── dataset/                        # 📁 Training dataset
├── requirements.txt                # 📦 Dependencies
├── README.md                       # 📖 This file
├── MOBILE_DEPLOYMENT_GUIDE.md     # 📱 Mobile guide
└── training_history.png           # 📊 Training visualization
```

## 🚀 Quick Start (Cleaned Version)

### Run the Working App
```bash
python brain_tumor_app.py
```

✅ **What works**: Mobile-optimized interface with public sharing
✅ **What works**: Optimized 0.570 threshold for accurate predictions  
✅ **What works**: Enhanced CNN model with 67% accuracy
✅ **What works**: Automatic public URLs for mobile access
   ```bash
   python improved_model_training.py
   ```

5. **Run the Gradio app:**
   ```bash
   python gradio_brain_tumor_app.py
   ```

### Option 2: Google Colab

1. **Upload files to Colab:**
   - Upload all Python files to your Colab environment
   - Upload your dataset or use the provided sample images

2. **Install dependencies:**
   ```python
   !pip install gradio tensorflow pillow matplotlib opencv-python
   ```

3. **Train the model (if needed):**
   ```python
   !python improved_model_training.py
   ```

4. **Run the app:**
   ```python
   !python gradio_brain_tumor_app.py
   ```

   The app will automatically detect Colab environment and enable public sharing.

## Dataset Structure

```
dataset/
├── yes/          # Images with brain tumors
│   ├── Y1.jpg
│   ├── Y2.jpg
│   └── ...
└── no/           # Images without brain tumors
    ├── N1.jpg
    ├── N2.jpg
    └── ...
```

## Model Architecture

The improved CNN model includes:

- **4 Convolutional Blocks** with BatchNormalization and Dropout
- **Progressive Feature Maps**: 32 → 64 → 128 → 256 filters
- **Data Augmentation**: Rotation, shifting, flipping, zooming
- **Advanced Callbacks**: Early stopping and learning rate reduction
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-score

## Usage Instructions

1. **Upload Image**: Click on the upload area and select a brain MRI scan
2. **Analyze**: Click the "🔍 Analyze Scan" button
3. **Review Results**: Check the prediction, confidence score, and visual overlay
4. **Try Examples**: Use the provided example images to test the system

## Files Description

- `gradio_brain_tumor_app.py` - Main Gradio application
- `improved_model_training.py` - Enhanced model training script
- `Brain_tumor_main.py` - Original Tkinter application (legacy)
- `braintumor_model_building.py` - Original model training (legacy)
- `requirements.txt` - Python dependencies
- `brain_tumor_model.h5` - Trained model file (generated after training)

## Important Notes

⚠️ **Medical Disclaimer**: This system is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Troubleshooting

### Common Issues:

1. **Model not found error:**
   - Make sure you've trained the model first
   - Check if `brain_tumor_model.h5` or `improved_brain_tumor_model.h5` exists

2. **CUDA/GPU issues:**
   - The system works with both CPU and GPU
   - For CPU-only: `pip install tensorflow-cpu`

3. **Memory issues:**
   - Reduce batch size in training script
   - Use smaller image sizes if needed

4. **Gradio sharing issues:**
   - Check your internet connection
   - Try running without sharing first

## Performance Tips

- **For better accuracy**: Use more training data and longer training epochs
- **For faster inference**: Use smaller image sizes (but may reduce accuracy)
- **For GPU acceleration**: Install CUDA-compatible TensorFlow version

## Customization

You can customize the system by:

- Modifying the CNN architecture in `improved_model_training.py`
- Adjusting the Gradio interface in `gradio_brain_tumor_app.py`
- Adding new preprocessing techniques
- Implementing additional evaluation metrics

## Contributing

Feel free to contribute by:
- Improving the model architecture
- Adding new features to the interface
- Enhancing the preprocessing pipeline
- Adding more evaluation metrics

## License

This project is for educational purposes. Please respect medical data privacy and regulations when using real medical images.