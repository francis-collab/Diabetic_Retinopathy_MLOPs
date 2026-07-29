# Diabetic Retinopathy Severity Classification – Full MLOps Pipeline

End-to-end Machine Learning Cycle for **diabetic-retinopathy severity grading** from fundus (retinal) images.  
Builds on the earlier tabular Diabetes Risk Prediction use-case and extends it to **non-tabular (image) data**.

**Best model (from notebook):** Advanced CNN with MobileNetV2 transfer learning  
**Production weights:** Re-exported under TensorFlow 2.13 (same architecture) so the API loads reliably  
**Preprocessing (must match notebook):** RGB → resize **128×128** → normalize `/255.0`

---

## 1. Video Demo

> **YouTube link (camera-on demo of prediction + retraining):**  
> https://youtu.be/dQw4w9WgXcQ  
> *(Replace with your real recorded demo URL before final submission.)*

---

## 2. Live URL

Deployment is **Dockerized** (see below). Local endpoints after `docker-compose up`:

- API docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`
- Health: `http://localhost:8000/health`

---

## Project description

| Item | Detail |
|------|--------|
| Task | 5-class severity classification of diabetic retinopathy |
| Classes | `No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR` |
| Data | Kaggle-style fundus images (APTOS-like folder layout) |
| Best model | Advanced CNN (MobileNetV2 + custom head) – highest ROC-AUC in notebook (~0.94); production retrain ≈ 0.71 accuracy on held-out test subset |
| Serving | FastAPI + nginx load balancer + Streamlit |
| Retraining | Upload new labelled images → fine-tune the **same** pre-trained CNN |
| Load test | Locust (1 / 2 / 4 API containers) |

---

## Repository structure

```text
Diabetic_Retinopathy_MLOPs/
├── README.md
├── requirements.txt
├── requirements-docker.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── notebook/
│   └── diabetic_retinopathy_mlops.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── keras_compat.py
├── scripts/
│   └── train_best_model.py
├── data/
│   ├── train/<class>/*.png
│   ├── test/<class>/*.png
│   ├── retrain_buffer/
│   └── uploads/
├── models/
│   ├── best_model.keras
│   ├── label_encoder.pkl
│   └── metrics_summary.pkl
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
└── locust/
    └── locustfile.py
```

---

## Setup (local, without Docker)

```bash
git clone <Your_Repo_URL>
cd Diabetic_Retinopathy_MLOPs

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Confirm model artefacts
ls models/best_model.keras models/label_encoder.pkl
```

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

Full fundus dataset is not stored in GitHub (size). Download the APTOS-style images from Kaggle and place them under data/train/<class>/ and data/test/<class>/. A few sample images may be included for smoke tests.

## Docker deployment (recommended)

```bash
docker-compose up -d --scale api=1
```

Services:

| Service | Role | Port |
|---------|------|------|
| `api` | FastAPI model server (scalable) | internal 8000 |
| `lb` | nginx load balancer | **8000** (host) |
| `ui` | Streamlit | **8501** |

Scale API containers for load testing:

```bash
docker-compose up -d --scale api=2
docker-compose up -d --scale api=4
```

Stop:

```bash
docker-compose down
```

### Predict / retrain with Docker

1. Open http://localhost:8501  
2. **Predict** tab → upload a fundus image → Predict Severity  
3. **Upload & Retrain** tab → upload images + comma-separated labels → Upload to Retrain Buffer → Trigger Retraining  

Or via curl:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -F "file=@data/test/No_DR/No_DR_0003.png"

curl -X POST http://localhost:8000/upload_retrain \
  -F "files=@data/test/No_DR/No_DR_0003.png" \
  -F "files=@data/test/Mild/Mild_0000.png" \
  -F "labels=No_DR,Mild"

curl -X POST "http://localhost:8000/retrain?epochs=3"
```

---

## Functionalities

| Requirement | Implementation |
|-------------|----------------|
| **1. Model prediction (single image)** | Streamlit Predict tab or `POST /predict` |
| **2. Visualizations + ≥3 feature interpretations** | Streamlit Visualizations tab – class distribution + clinical interpretations |
| **3. Upload bulk data for retrain** | `POST /upload_retrain` → saved under `data/retrain_buffer/<label>/` |
| **4. Trigger retraining** | Button / `POST /retrain` – fine-tunes the existing Advanced CNN |
| **Model uptime** | Sidebar + `GET /health` |
| **Locust flood simulation** | `locust/locustfile.py` (results below) |

---

## Retraining flow

1. User uploads one or more fundus images + matching labels (`No_DR,Mild,...`).  
2. Images are **saved to disk** under `data/retrain_buffer/<label>/` (acts as the “database”).  
3. On “Trigger Retraining”:  
   - Load original `data/train` images (same 128×128 preprocessing).  
   - Load buffer images.  
   - Concatenate → label-encode.  
   - Load the **existing** `best_model.keras` (Advanced CNN / MobileNetV2).  
   - **Fine-tune** a few epochs with class weights + early stopping.  
   - Save updated model & encoder → hot-reload into the API.

---

## Load testing with Locust

```bash
# API stack must be running
locust -f locust/locustfile.py --host http://localhost:8000 \
  --headless -u 20 -r 5 -t 60s --csv=results_Ncontainers
```

### Results from Flood Request Simulation

Load profile: **20 concurrent users**, spawn rate 5/s, duration **60 s**, targeting `/predict` (and light `/health`).

| Containers | Users | Predict reqs | Predict RPS | Median latency (ms) | 95% latency (ms) | Predict failures |
|------------|-------|--------------|-------------|---------------------|------------------|------------------|
| 1          | 20    | 703          | ~11.8       | **760**             | **1200**         | **0**            |
| 2          | 20    | 716          | ~12.1       | **760**             | **1400**         | **0**            |
| 4          | 20    | 721          | ~12.1       | **780**             | **1200**         | **0**            |

**Observation:** With a CPU-only TensorFlow inference workload, adding containers reduces health-check timeouts (failures drop to zero at 2+ replicas) and slightly increases successful throughput. Median predict latency stays in the same band because the host CPU is the bottleneck; further gains would require GPU or a larger machine. This is the expected pattern for heavy model inference under Locust.

---

## Notebook summary

The notebook (`notebook/diabetic_retinopathy_mlops.ipynb`) contains:

1. Data acquisition & exploratory analysis (class imbalance story).  
2. Preprocessing (128×128, /255).  
3. **6 experiments**: RandomForest, LogisticRegression, SVM, XGBoost, Simple CNN, **Advanced CNN (MobileNetV2)**.  
4. Optimization: class_weight, L2, dropout, data augmentation, early stopping, ReduceLROnPlateau, transfer learning.  
5. Metrics per experiment: Accuracy, Precision, Recall, F1, ROC-AUC (OVR).  
6. Plots: confusion matrices, learning curves, ROC curves.  
7. Best model selected by ROC-AUC → Advanced CNN.

Production `models/best_model.keras` uses the **same MobileNetV2 architecture**, trained/exported under TensorFlow 2.13 for reliable serving.

---

## Dataset insights (Visualizations tab)

1. **Severity imbalance** – Majority of images are No_DR / Mild, matching real screening populations; handled with `class_weight='balanced'` and transfer learning.  
2. **Lesion patterns** – Higher severity classes show more hemorrhages, exudates and neovascularization; CNN features capture these markers.  
3. **Clinical link** – The earlier tabular model predicted *risk of diabetes*; this image model detects an actual *complication* (retinopathy). Together: risk score → fundus photo → severity grade → referral.

---

## Author

**Francis Mutabazi** – aspiring Machine Learning Engineer
