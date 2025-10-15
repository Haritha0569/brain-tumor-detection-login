# 🧠 Brain Tumor Detection with Login System

An AI-powered **Brain Tumor Detection Web App** built using **Gradio**, **Hugging Face Transformers**, and **PyTorch**, featuring a secure **SQLite-based Login & Registration System**.  


## 🚀 Features

✅ **User Authentication** – Login and registration system using SQLite  
✅ **AI Model Integration** – Brain tumor detection using Hugging Face model  
✅ **Interactive UI** – Built with Gradio for easy MRI upload and prediction  
✅ **Confidence Score Display** – Shows how confident the model is  
✅ **Lightweight & Easy to Run** – Works locally or on Google Colab  


## 🧩 Tech Stack

- **Python 3.11+**
- **PyTorch**
- **Hugging Face Transformers**
- **Gradio**
- **SQLite**



## 📸 App Overview

### 🔐 Login / Register Page
Users can create an account or log in securely to access the tumor detection system.

### 🧠 MRI Upload Page
Upload an MRI scan and let the AI model predict whether a **Tumor** or **No Tumor** is detected.



## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/brain-tumor-detection-with-login.git
cd brain-tumor-detection-with-login


2️⃣ Install dependencies
pip install torch torchvision transformers gradio

3️⃣ Run the app
python app.py


Gradio will provide a local and public link — open it in your browser.

🧠 Model Used

Model: ShimaGh/Brain-Tumor-Detection

Base Framework: Hugging Face Transformers

Output Classes:

Tumor

No Tumor

🗄️ Database

Uses SQLite (users.db) to store user credentials.

For simplicity, passwords are stored as plain text —
👉 In production, use hashing (SHA256 / bcrypt) for better security.

📈 Future Enhancements

🔐 Password encryption

📧 Email verification system

🧠 Model performance improvement with fine-tuning

☁️ Cloud deployment (Hugging Face Spaces / Streamlit Cloud / Render)

