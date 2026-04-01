# DeepCarAnalysis - AI-Powered Vehicle Detection & Classification System

🚗 **DeepCarAnalysis** is a production-ready Streamlit web application for vehicle image analysis with two independent AI pipelines: CNN-based Image Classification and YOLO-based Object Detection.

## 🎯 Features

### CNN-Based Image Classification
- **Models Supported**: VGG16, VGG19, ResNet50, DenseNet121, MobileNetV2
- **Task**: Car damage severity classification (Minor, Moderate, Severe)
- **Explainable AI**: Grad-CAM, Grad-CAM++, Eigen-CAM support

### YOLO-Based Object Detection
- **Models Supported**: YOLOv8, YOLOv11, YOLOv12
- **Task**: Car damage detection and localization
- **Features**: Adjustable confidence threshold, bounding box visualization

### UI/UX Features
- **Modern Interface**: Clean, professional design with wide layout
- **Model Selection**: Dynamic dropdown loading available models
- **Image Input**: Upload images or select from bundled samples
- **Real-time Analysis**: Instant predictions with visual feedback
- **Export Options**: Download analysis results and visualizations

## 🏗️ Project Structure

```
DeepCarAnalysis/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── models/                  # Directory for model files (if needed)
├── utils/                   # Utility modules
│   ├── preprocessing.py     # Image preprocessing functions
│   ├── inference.py         # Model loading and inference
│   ├── xai.py              # Explainable AI implementations
│   └── yolo_utils.py       # YOLO-specific utilities
├── assets/                  # Static assets (sample images, etc.)
└── Models_Weight_files/     # Pre-trained model weights
    ├── vgg16_best.h5
    ├── vgg19_best.h5
    ├── resnet50_best.h5
    ├── densenet121_best.h5
    ├── mobilenetv2_best.h5
    ├── bestyolov8.pt
    ├── best(1)yolov11.pt
    └── best(1)yolov12.pt
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or download the project**:
   ```bash
   cd /path/to/your/project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model weights are in place**:
   - Place your trained model weights in the `Models_Weight_files/` directory
   - Supported formats: `.h5` for CNN models, `.pt` for YOLO models

## 🎮 Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - Open your browser and navigate to `http://localhost:8501`

### Using the Application

1. **Select Analysis Type**:
   - Choose between "CNN Classification" or "YOLO Detection"

2. **Choose a Model**:
   - Select from available trained models in the sidebar

3. **Upload or Select Image**:
   - Upload your own image (JPG/PNG)
   - Or select from provided sample images

4. **View Results**:
   - Classification: See predicted damage severity with confidence scores
   - Detection: View bounding boxes and detected damage locations

5. **Explore XAI (Classification only)**:
   - Enable "Show Explainable AI" checkbox
   - Select explanation methods (Grad-CAM, Grad-CAM++, Eigen-CAM)
   - View heatmaps and overlays

## 🔧 Configuration

### Adding New Models

1. **CNN Models**: Place `.h5` files in `Models_Weight_files/`
2. **YOLO Models**: Place `.pt` files in `Models_Weight_files/`
3. **Sample Images**: Add images to `assets/` directory

### Model Requirements

**CNN Classification Models**:
- Input shape: (224, 224, 3)
- Output: 3 classes (Minor, Moderate, Severe)
- Framework: TensorFlow/Keras

**YOLO Detection Models**:
- Input size: 640x640
- Framework: Ultralytics YOLO
- Classes: Car damage types (scratches, dents, etc.)

## 📊 Model Information

### CNN Classification Models
| Model | Architecture | Input Size | Parameters | Use Case |
|-------|-------------|------------|------------|----------|
| VGG16 | VGG16 + FC | 224x224 | ~138M | General classification |
| VGG19 | VGG19 + FC | 224x224 | ~144M | Detailed features |
| ResNet50 | Residual Network | 224x224 | ~26M | Deep features |
| DenseNet121 | Dense Connections | 224x224 | ~8M | Parameter efficient |
| MobileNetV2 | Depthwise Separable | 224x224 | ~3.5M | Mobile deployment |

### YOLO Detection Models
| Model | Architecture | Input Size | Use Case |
|-------|-------------|------------|----------|
| YOLOv8 | CSPDarknet | 640x640 | Fast detection |
| YOLOv11 | Improved CSP | 640x640 | Better accuracy |
| YOLOv12 | Latest YOLO | 640x640 | State-of-the-art |

## 🎨 Explainable AI (XAI)

The application implements advanced XAI techniques for CNN models:

- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Grad-CAM++**: Improved version with better localization
- **Eigen-CAM**: Eigenvalue-based activation mapping

These methods help understand which parts of the image contribute most to the model's predictions.

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure model files are in the correct directory
   - Check file permissions
   - Verify model compatibility

2. **CUDA/GPU Issues**:
   - Install CUDA-compatible versions
   - Set `CUDA_VISIBLE_DEVICES=""` for CPU-only mode

3. **Memory Errors**:
   - Reduce batch size in preprocessing
   - Use smaller input sizes
   - Restart the application

4. **Import Errors**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

### Performance Optimization

- Use GPU for faster inference
- Pre-load models to reduce loading time
- Optimize image preprocessing pipeline

## 📈 Performance Metrics

The system leverages advanced deep learning architectures to ensure robust performance across diverse conditions.

### 🚗 Damage Severity Classification
| Model Architecture | Accuracy | Role |
| :--- | :--- | :--- |
| **Ensemble (Stacking)** | **~96.0%** | **Final Combined Model** |
| DenseNet121 | 67.5% | Base Classifier |
| ResNet50 | 66.0% | Base Classifier |
| VGG19 | 66.0% | Base Classifier |
| MobileNetV2 | 65.4% | Base Classifier |
| VGG16 | 65.0% | Base Classifier |

### 🛠️ Damage Detection (Localization)
| Detection Model | mAP@0.5 | Performance |
| :--- | :--- | :--- |
| **YOLOv8** | **71.5%** | **Recommended** |
| YOLOv11 | 70.8% | High Accuracy |
| YOLOv12 | 68.4% | Cutting Edge |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- Ultralytics for YOLO implementations
- Streamlit for the web application framework
- CarDD dataset for training data

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review the code documentation
- Open an issue on GitHub

---

**Built with ❤️ for AI-powered vehicle analysis**
