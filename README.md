# 🎓 CARRO 

An explainable, counselling-aware early warning system to predict student dropout risk and suggest targeted interventions. Built for Smart India Hackathon 2025.

---

## 🚀 Features

- 🔍 **Risk Scoring Engine** – Calibrated logistic regression with feature-level impact analysis
- 🧠 **Counselling-Aware Inputs** – Includes emotional and behavioral indicators from counselling logs
- 📊 **Explainability** – Top 5 contributing features per student, with direction and impact
- ⚖️ **Fairness Audit** – Evaluates model bias across student groups (e.g., urban vs rural)
- 🖥️ **Streamlit UI** – Interactive sliders, live risk score, and intervention suggestions
- 🧪 **Synthetic Data Generator** – Realistic student profiles with dropout labels
- 🔌 **FastAPI Service** – Real-time scoring via REST API
- 🧰 **Reproducible Build** – One-command setup with version-agnostic compatibility
- 📜 **MIT License** – Clear, permissive licensing for demo and reuse

---

## 📂 Project Structure
├── data/ # Synthetic dataset 
├── model/ # Trained model, scaler, metrics 
├── src/ 
│ ├── generate_data.py # Creates student_term.csv 
│ ├── train.py # Trains model and saves artifacts 
│ ├── service.py # FastAPI scoring service 
│ ├── demo_app.py # Streamlit UI 
│ ├── fairness_audit.py # Bias analysis
│ └── utils.py  #Shared functions #Python dependencies
├── requirements.txt  #Python dependencies
├── LICENSE  #MIT License
└── README.md 


---
 ⚙️ Setup & Build
 1. Clone the repo :
    ```bash
git clone https://github.com/your-username/dropout-predictor.git
cd dropout-predictor
   
 2. Create and activate virtual environment :
    python -m venv .venv
    source .venv/Scripts/activate  # Windows Git Bash

 3. Install dependencies :
    pip install -r requirements.txt

 4. Build the system (generate data + train model) :
    python src/generate_data.py && python src/train.py


🧪 Run the Demo
🔌 Start the API
 python -m uvicorn src.service:app --reload --host 0.0.0.0 --port 8000


 🖥️ Launch the UI : 
 python -m streamlit run src/demo_app.py


 📊 Fairness Audit (Optional) :
 python src/fairness_audit.py


📜 License
 This project is licensed under the MIT License. You are free to use, modify, and distribute this software with proper attribution.



🙌 Acknowledgements
Developed by Team Urecon.
Project Name: CARRO – Counselling-Aware Risk and Retention Optimizer
Smart India Hackathon 2025 – Software Edition
