#  Health Insurance Premium Predictor

A complete machine learning web application with a **FastAPI** backend and a **Streamlit** frontend to predict health insurance premium risk categories (**Low**, **Medium**, or **High**).

---

##  Quick Start

### 1. Activate Virtual Environment
```bash
source myenv/bin/activate
```

### 2. Run the FastAPI Backend
Start the FastAPI server on port 8000:
```bash
uvicorn app:app --reload
```
- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Prediction endpoint: `POST http://127.0.0.1:8000/predict`

### 3. Run the Streamlit Frontend
In a new terminal window (with the virtual environment activated):
```bash
streamlit run streamlit_app.py
```
Or with python executable directly:
```bash
./myenv/bin/python -m streamlit run streamlit_app.py
```

---

##  Features

- **Interactive UI**: Sliders, number inputs, city search, and occupation selectors with instant input validation.
- **Real-Time Health Metric Calculations**: Automatically calculates BMI, BMI classification, Age Group, City Tier, and Lifestyle Risk according to Pydantic computed field rules.
- **Dual Prediction Modes**:
  1. **FastAPI Server (REST API)**: Sends asynchronous JSON payloads to the `/predict` endpoint.
  2. **Direct ML Model Standalone**: Loads `model.pkl` directly within Streamlit for offline testing without running the backend server.
- **Quick-Load Sample Profiles**: 1-click presets ("Healthy Young Student", "Middle-Aged Corporate Professional", "Senior Smoker Business Owner") to easily test different risk scenarios.
- **Probability Breakdown**: Visual bar chart of model class probabilities (`Low`, `Medium`, `High`).
- **Developer Debug Tools**: Expandable drawer showing full JSON payload, derived features, and equivalent `curl` command.

---

##  Project Structure

```text
├── app.py                # FastAPI REST API with Pydantic validation & ML prediction endpoint
├── streamlit_app.py      # Streamlit web interface
├── model.pkl             # Trained scikit-learn Pipeline (RandomForestClassifier)
├── insurance.csv         # Dataset used for training
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
```
