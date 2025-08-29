# 🎨 Neural Style Transfer CV App

A **Neural Style Transfer Web App** built with **PyTorch** and **Streamlit**.  
Upload your own images and transform them into **artistic masterpieces** by blending content and style images together.

---

## 🚀 Features
- 🖼 Upload **content image** + **style image**
- 🎨 Apply **Neural Style Transfer** using PyTorch
- ⚡ Fast and interactive **Streamlit Web App**
- 💾 Download the generated **output image**
- 📓 Comes with a **Jupyter Notebook** for experimentation

---

## 📂 Project Structure
```
📁 Neural-Style-Transfer-CV-App
│── 1.jpg                 # Sample content image
│── 2.jpg                 # Sample style image
│── OUTPUT.png            # Example output
│── ST APP.py             # Streamlit Web App
│── ST JP NB.ipynb        # Style Transfer Notebook
│── ST JP NB 2.ipynb      # Extra Notebook
│── README.md             # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/Pranitttt64/Neural-Style-Transfer-CV-App.git
   cd Neural-Style-Transfer-CV-App
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   venv\Scripts\activate       # On Windows
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Run Streamlit App
```bash
streamlit run "ST APP.py"
```

- Upload **content** and **style** images
- Wait for model to generate stylized output
- Save the result 🎉

### Run Notebook
Open **Jupyter Notebook** and explore:
```bash
jupyter notebook "ST JP NB.ipynb"
```

---

## 📸 Example

| Content | Style | Output |
|---------|-------|--------|
| ![Content](2.jpg) | ![Style](1.jpg) | ![Output](OUTPUT.png) |

---

## 🛠 Tech Stack
- **Python**
- **PyTorch**
- **Streamlit**
- **Jupyter Notebook**
- **OpenCV, PIL, NumPy**

---

## 🌟 Future Improvements
- [ ] Add **multiple styles blending**
- [ ] Deploy on **Streamlit Cloud / Hugging Face Spaces**
- [ ] Add **GPU acceleration option**

---

## 🤝 Contributing
Pull requests are welcome!  
If you’d like to suggest features, open an **issue**.

---

## 📜 License
This project is licensed under the MIT License.

---

## 👨‍💻 Author
**Pranit Saundankar**  
🔗 [GitHub Profile](https://github.com/Pranitttt64)
