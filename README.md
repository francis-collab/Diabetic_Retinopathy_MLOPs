# Diabetic Retinopathy Severity Classification – Full MLOps Pipeline

End-to-end Machine Learning Cycle for **diabetic-retinopathy severity grading** from fundus (retinal) images.  
Builds on the earlier tabular Diabetes Risk Prediction use-case and extends it to **non-tabular (image) data**.

**Best model (from notebook):** Advanced CNN with MobileNetV2 transfer learning  
**Preprocessing (must match notebook):** RGB → resize **128×128** → normalize `/255.0`

---

## 1. Video Demo
> **YouTube link (camera-on demo of prediction + retraining):**  
> `https://youtu.be/YOUR_VIDEO_ID`  
> *(Replace with your real video URL after recording.)*

## 2. Live URL (if deployed)
- API docs: `http://<your-cloud-ip>:8000/docs`
- Streamlit UI: `http://<your-cloud-ip>:8501`

---

## Project description

| Item | Detail |
|------|--------|
| Task | 5-class severity classification of diabetic retinopathy |
| Classes | `No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR` |
| Data | Kaggle-style fundus images (APTOS-like folder layout) |
| Best model | Advanced CNN (MobileNetV2 + custom head) – highest ROC-AUC ≈ 0.94 |
| Serving | FastAPI + Streamlit |
| Retraining | Upload new labelled images → fine-tune the **same** pre-trained CNN |
| Load test | Locust |

---

## Repository structure

```text
Diabetic_Retinopathy_MLOPs2/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── notebook/
│   └── diabetic_retinopathy_mlops.ipynb   
├── src/
│   ├── preprocessing.py   # IMG_SIZE=128, same as notebook
│   ├── model.py           # classical ML + CNN + fine_tune_keras_model
│   └── prediction.py      # single-image predict using best Keras model
├── scripts/
│   └── train_best_model.py
├── data/
│   ├── train/<class>/*.png
│   ├── test/<class>/*.png
│   ├── retrain_buffer/    # uploaded images for retraining
│   └── uploads/
├── models/
│   ├── best_model.keras   # production Advanced CNN
│   ├── label_encoder.pkl
│   └── metrics_summary.pkl
├── api/
│   └── main.py            # FastAPI: /predict /upload_retrain /retrain /health …
├── app/
│   └── streamlit_app.py   # UI: uptime, viz, predict, upload+retrain
└── locust/
    └── locustfile.py
```

---

## Setup (local)

```bash
# 1. Clone the repo and switch to it
git clone < Your_Repo_URL >
cd Diabetic_Retinopathy_MLOPs

# 2. Create virtual environment (Python 3.10 recommended)
python3.10 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Confirm the best model is present
ls models/best_model.keras models/label_encoder.pkl
```

### Start the services

**Terminal 1 – API**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 – Streamlit UI**
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

- API docs: http://localhost:8000/docs  
- UI: http://localhost:8501  

---

## Docker (recommended for deployment & load testing)

```bash
docker compose up --build
```

This starts:
- `api` on port 8000
- `ui`  on port 8501 (talks to api via Docker network)

Scale API containers for load-test comparison:
```bash
docker compose up --build --scale api=2
# or --scale api=4
```

---

## Functionalities

| Requirement | How it is implemented |
|-------------|------------------------|
| **1. Model prediction (single image)** | Streamlit “Predict” tab or `POST /predict` with an image file |
| **2. Visualizations + ≥3 feature interpretations** | Streamlit “Visualizations” tab – class distribution (train/test/buffer) + written clinical interpretations |
| **3. Upload bulk data for retrain** | Streamlit “Upload & Retrain” or `POST /upload_retrain` (multiple images + comma-separated labels) → saved under `data/retrain_buffer/<label>/` |
| **4. Trigger retraining** | Button “Trigger Retraining” or `POST /retrain` – **fine-tunes the existing Advanced CNN** (custom pre-trained model) on original + new data |
| **Model uptime** | Sidebar of Streamlit + `GET /health` |
| **Locust flood simulation** | `locust/locustfile.py` (see section below) |
| **Notebook** | 6 experiments, regularization / early stopping / class weights / transfer learning, ≥4 metrics, confusion matrices, learning curves, ROC curves. Best model selected by ROC-AUC. **Do not edit.** |

---

## Retraining flow 

1. User uploads one or more fundus images + matching labels (`No_DR,Mild,...`).
2. Images are **saved to disk** under `data/retrain_buffer/<label>/` (acts as the “database”).
3. On “Trigger Retraining”:
   - Load original `data/train` images (same 128×128 preprocessing as notebook).
   - Load buffer images.
   - Concatenate → label-encode.
   - Load the **existing** `best_model.keras` (Advanced CNN / MobileNetV2).
   - **Fine-tune** a few epochs with class weights + early stopping.
   - Save updated model & encoder → hot-reload into the API.

---

## Load testing with Locust

```bash
# API must be running (local or Docker)
locust -f locust/locustfile.py --host http://localhost:8000
```

Headless example (record latency / response time):
```bash
# 1 container
locust --headless -u 20 -r 5 -t 60s -f locust/locustfile.py --host http://localhost:8000 --csv=results_1container

# After scaling to 2 / 4 containers (docker compose --scale api=N)
locust --headless -u 20 -r 5 -t 60s -f locust/locustfile.py --host http://localhost:8000 --csv=results_Ncontainers
```

**Expected observation:** median and 95th-percentile latency drop (or throughput rises) as the number of API containers increases, until the host CPU/GPU is saturated.

Paste screenshots / CSV summary tables into this README under “Results from Flood Request Simulation”.

---

## Cloud deployment (example – any platform)

### Option A – single VM (AWS EC2 / GCP / Azure / DigitalOcean)

```bash
# on the VM
sudo apt update && sudo apt install -y docker.io docker-compose git
git clone <your-repo-url>
cd Diabetic_Retinopathy_MLOPs
docker compose up --build -d
```

Open security group / firewall for ports 8000 and 8501.

### Option B – Render / Railway / Hugging Face Spaces
- Point the service to the Dockerfile (or use the `uvicorn` / `streamlit` commands).
- Mount / upload the `models/` folder so `best_model.keras` is present at runtime.

After deployment, record the public URL in the “Live URL” section above and demonstrate evaluation in production via the Streamlit UI + `/metrics` endpoint.

---

## Notebook summary 

The notebook (`notebook/diabetic_retinopathy_mlops.ipynb`) contains:

1. Data acquisition & exploratory analysis (class imbalance story).
2. Preprocessing (128×128, /255).
3. **6 experiments**: RandomForest, LogisticRegression, SVM, XGBoost, Simple CNN, **Advanced CNN (MobileNetV2)**.
4. Optimization techniques: class_weight, L2, dropout, data augmentation, early stopping, ReduceLROnPlateau, transfer learning.
5. Metrics per experiment: Accuracy, Precision, Recall, F1, ROC-AUC (OVR).
6. Plots: confusion matrices, learning curves, ROC curves.
7. Best model selected by ROC-AUC → saved as `models/best_model.keras` (+ label encoder + metrics summary).

---

## Quick API test (after services are up)

```bash
# Health
curl http://localhost:8000/health

# Predict (replace with a real fundus image)
curl -X POST http://localhost:8000/predict \
  -F "file=@data/test/No_DR/No_DR_0003.png"

# Dataset stats
curl http://localhost:8000/dataset_stats
```

---

## Results from Flood Request Simulation

*(Fill after you run Locust – example table)*

| Containers | Users | RPS | Median latency (ms) | 95% latency (ms) | Failures |
|------------|-------|-----|---------------------|------------------|----------|
| 1          | 20    | …   | …                   | …                | 0        |
| 2          | 20    | …   | …                   | …                | 0        |
| 4          | 20    | …   | …                   | …                | 0        |

---

## Author notes

- The production prediction path uses **exactly the same preprocessing** as the notebook (RGB, 128×128, /255).
- Retraining fine-tunes the **same** Advanced CNN that was selected as best in the notebook.

Done by **Francis Mutabazi** , an aspiring Machine Learning Engineer