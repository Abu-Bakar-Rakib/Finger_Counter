# Finger Counter

A real-time hand gesture recognition system that detects and counts the number of fingers shown to the camera using computer vision and machine learning techniques.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project implements a real-time finger counting system using MediaPipe's advanced hand tracking model and OpenCV for image processing. The application processes video frames from your webcam to identify hand landmarks and accurately determine the number of extended fingers.

**Key Capabilities:**
- Real-time hand detection and tracking
- Accurate finger counting algorithm
- Low-latency processing suitable for interactive applications
- Simple command-line interface

---

## ✨ Features

- ✅ **Real-Time Processing**: Processes video frames from webcam with minimal latency
- ✅ **Accurate Detection**: Utilizes MediaPipe's state-of-the-art hand tracking model
- ✅ **Efficient Algorithm**: Lightweight implementation optimized for CPU performance
- ✅ **Visual Feedback**: Displays hand landmarks and finger count on video stream
- ✅ **Easy Integration**: Simple Python API for integration into other projects
- ✅ **Configurable Confidence**: Adjustable detection and tracking confidence thresholds

---

## 🛠️ Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **OpenCV** | 4.5+ | Image processing and video capture |
| **MediaPipe** | 0.8+ | Hand detection and tracking |
| **NumPy** | Latest | Numerical computations |

---

## 📦 Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- 100 MB available disk space
- 2 GB RAM minimum

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Abu-Bakar-Rakib/Finger_Counter.git
cd Finger_Counter
```

### Step 2: Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

### Step 3: Verify Installation

```bash
python -c "import cv2, mediapipe; print('Installation successful!')"
```

---

## 💻 Usage

### Basic Usage

Run the finger counter application:

```bash
python finger_counter.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| Any other key | Continue running |

### Output

The application displays:
- Live video feed from webcam
- Hand skeleton overlay with joint points
- Current finger count in the top-left corner

---

## 📁 Project Structure

```
Finger_Counter/
├── finger_counter.py      # Main application script
├── finger_count.jpg       # Sample output image
├── README.md             # Project documentation
└── .gitignore           # Git ignore configuration
```

---

## 🔧 How It Works

### Algorithm Overview

1. **Hand Detection**: MediaPipe detects hand presence and position in the frame
2. **Landmark Extraction**: 21 hand landmarks are identified for each detected hand
3. **Finger Analysis**: Each finger is analyzed based on landmark positions:
   - **Thumb**: Detected by comparing x-coordinates of thumb tip and joint
   - **Index to Pinky**: Detected by comparing y-coordinates of fingertip and joint
4. **Counting**: Extended fingers are counted and displayed in real-time

### Key Components

```python
count_fingers(hand_landmarks)
```

Analyzes hand landmarks and returns the number of extended fingers (0-5).

**Parameters:**
- `hand_landmarks`: MediaPipe hand landmark object

**Returns:**
- `int`: Number of fingers extended (0-5)

---

## 🎓 Example Output

```
Finger Count: 3
Hand Landmarks: [21 joints detected]
Confidence: 0.95
```

---

## 🚧 Future Enhancements

- [ ] **Multi-Hand Support**: Detect and count fingers in both hands simultaneously
- [ ] **Gesture Recognition**: Identify specific hand gestures (peace, thumbs up, etc.)
- [ ] **Sign Language Integration**: Translate hand gestures to sign language
- [ ] **Performance Optimization**: GPU acceleration support
- [ ] **Mobile Deployment**: Cross-platform mobile application
- [ ] **Data Logging**: Record and analyze finger counting statistics
- [ ] **Gesture-Based Controls**: Control applications through hand gestures
- [ ] **Web Interface**: Browser-based application using Flask/FastAPI

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows Python best practices and includes appropriate comments.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact & Support

**Author:** Abu Bakar Rakib

- 📧 **Email**: [rakibcdp@gmail.com](mailto:rakibcdp@gmail.com)
- 🐙 **GitHub**: [@Abu-Bakar-Rakib](https://github.com/Abu-Bakar-Rakib)
- 💼 **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/abu-bakar-rakib)

For issues, questions, or suggestions, please open an issue on the [GitHub Issues](https://github.com/Abu-Bakar-Rakib/Finger_Counter/issues) page.

---

## 📚 References

- [MediaPipe Documentation](https://mediapipe.dev/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Hand Gesture Recognition Research](https://arxiv.org/search/?query=hand+gesture+recognition)

---

**Last Updated:** June 2, 2026  
**Version:** 1.0.0
