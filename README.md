# 🧠 NeuLife — Acoustic Neurodegenerative Risk Profiling

This repository hosts an end-to-end machine learning system designed to analyze human voice patterns for signs of neurodegenerative conditions, specifically Parkinson's disease. Using a dataset of acoustic vocal biomarkers, this project implements a comparative predictive pipeline, maps feature importance using model explainability tools, and provides a live, browser-based microphone inference loop.

---

## 🛠️ System Architecture & Workflow

The core engineering pipeline is divided into six distinct phases:

1. **Data Integrity & Stratified Splitting:** Isolated training ($80\%$) and testing ($20\%$) partitions using stratified splits to handle class imbalances, ensuring the patient-to-healthy ratio is perfectly mirrored across both sets.
2. **Data Leakage Prevention:** Enforced strict feature isolation by fitting the transformation scaler exclusively on the training partition and transforming the test metrics using memorized training constants.
3. **Multi-Model Benchmarking:** Evaluated three distinct algorithmic architectures to compare decision boundary strategies: Random Forest (Ensemble Learning), Support Vector Machines (Margin-Based Classification), and XGBoost (Gradient Boosting).
4. **Clinical Metric Optimization:** Prioritized **Recall (Sensitivity)** over global accuracy to minimize the clinical risk of False Negatives (unhandled patient cases). The optimized SVM baseline achieved a perfect **100% Recall** on the test partition.
5. **Model Explainability (SHAP):** Integrated **SHAP (SHapley Additive exPlanations)** to map the model's internal logic. The feature attribution matrix proved that Pitch Period Entropy (`PPE`) and frequency variance (`spread1`) are the primary indicators driving high-risk diagnoses.
6. **Production Latency Profiling:** Benchmarked system execution speeds over 1,000 continuous loops to measure deployment viability. The SVM model achieved a single-vector inference latency of just **0.196 ms**, making it ideal for edge execution.

---

## 🎙️ Live In-Memory Stream Interface

The repository features a built-in, real-time diagnostic simulation inside the notebook environment that completely bypasses disk storage:
* **Zero Disk Footprint:** Captures browser microphone streams via JavaScript WebRTC fragments and passes the binary chunks directly into Python's memory buffer.
* **Native Digital Signal Processing:** Automatically processes the raw audio wave using a native **NumPy Discrete Fourier Transform (DFT)** matrix to track cycle-to-cycle frequency shifts and execute immediate model inference.

---

## 🎛️ Tech Stack
* **Language:** Python 3 & JavaScript (WebRTC)
* **Frameworks:** Scikit-Learn, XGBoost, SHAP, NumPy, SciPy, PyDub
