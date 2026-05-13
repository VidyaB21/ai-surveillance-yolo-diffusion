# 🔍 AI Surveillance System using YOLO and Diffusion Models

An AI-powered surveillance system that integrates computer vision and generative AI to perform real-time object detection, tracking, behavior analysis, and scene reconstruction from video input.

---

## 🚀 Features

- 🎯 **Object Detection** using YOLOv8  
- 🔄 **Multi-Object Tracking** with unique IDs  
- ⚠️ **Behavior Analysis** (detects suspicious movement like running)  
- 🧠 **AI Scene Generation** using diffusion models  
- 🖼️ **Automatic Scene Reconstruction** from detected objects  
- 📊 **Logging System** for detected events  
- 🖥️ **Streamlit UI** for easy interaction  

---

## 🧠 Tech Stack

- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- PyTorch  
- Diffusers (Stable Diffusion)  
- Streamlit  

---

## 📂 Project Structure
ai-surveillance-system/
│
├── app/
│ ├── main_pipeline.py
│ ├── ui.py
│ ├── scene_generator.py (optional)
│
├── data/
├── outputs/
├── .gitignore
├── requirements.txt
└── README.md

---

## 📸 Output

- Generated AI images are saved in:
- outputs/
- 
- Logs are saved in:
- outputs/log.txt
- 
---

## ⚠️ Notes

- First run may take time due to model loading  
- Diffusion model may require higher RAM/GPU for faster performance  
- API-based generation is optional (local generation recommended)  

---

## 🧠 How It Works

1. Video input is processed frame-by-frame  
2. YOLO detects and tracks objects  
3. Movement is analyzed to detect suspicious behavior  
4. Detected objects are converted into prompts  
5. Diffusion model generates reconstructed AI scenes  

---

## 🎯 Use Cases

- Smart surveillance systems  
- Security monitoring  
- Behavior analysis in public areas  
- AI-based video understanding  

---

## 📌 Future Improvements

- Real-time streaming support  
- Advanced anomaly detection  
- Custom-trained YOLO models  
- Cloud deployment  

---

## 👨‍💻 Author

Vidya Bag  

---


