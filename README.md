## 🧬 Probabilistic Ancestral Inference Research Project

This repository contains a research framework designed to reconstruct latent states within high-dimensional, hierarchical stochastic datasets. While the project utilizes biological rules for data generation, its primary purpose is benchmarking the robustness of various probabilistic machine-learning architectures under controlled data degradation.

---

### 🌟 Project Goal

The core mission is to evaluate the technical trade-offs between computational efficiency and inference precision. Key objectives include:

#### 🛠️ Key Components

* **Hierarchical Modeling:** Designing a system to model latent dependencies within stochastic data.
* **Multi-Model Benchmarking:** Implementing and comparing Bayesian Inference, Hidden Markov Models (HMM), and Graph Neural Networks (GNN)..
* **Quantitative Validation:** Using "ground-truth" data generators to measure recovery performance against systematic masking.
* **Visualization:** A React dashboard to visualize the data and models.

---

### 🛠 Technical Stack & Reproducibility

The project emphasizes engineering rigor and reproducibility through a modular architecture:

* **Data Engine:** Powered by `msprime` for high-fidelity simulation of multi-generational datasets.
* **Meta-Replay System:** Uses JSON-based metadata and specific random seeds to ensure exact dataset reconstruction for benchmarking.
* **Inference Frameworks:**
    * **Bayesian:** Developed via `PyMC` and `PyTensor`
    * **HMM:** Implemented using `pomegranate` or `hmmlearn`
    * **GNN:** Built with PyTorch `Geometric` for multi-way dependency modeling
* **Application Layer:** A `FastAPI` backend serving a custom `React` dashboard for interactive visualization of calibration and uncertainty.
* **Deployment:** Fully containerized with `Docker` for cross-platform portability.

---

### 📊 Evaluation Metrics

Models are benchmarked using both quantitative and qualitative dimensions:


* **Reconstruction Accuracy:** Precision, recall, and F1-scores compared against the known "truth".
* **Model Calibration:** Aligning reported confidence intervals with actual recovery rates.
* **Computational Robustness:** Measuring the "break point" of each architecture across a spectrum of masking rates.
* **Statistical Significance:** Validation via chi-square and likelihood-ratio tests.

---

### 🚀 Getting Started

The entire system will be containerized with Docker to ensure reproducibility and ease of deployment.

#### How to Run Locally

You can build and run the complete Proof of Concept environment using `docker-compose`.

```bash
docker compose build
docker compose up
```

Once the containers are running, the Streamlit dashboard will be accessible via your web browser.

#### How to Rebuild Containers (If you mess them up)
To ensure a clean environment, you can stop, remove, and rebuild your Docker containers:

1. Stop and Remove Running Containers:
```bash
docker compose down
```

2. Force Rebuild Images (Pulls fresh dependencies):

```bash
docker compose build --no-cache
```

3. Restart the Service:
```bash
docker compose up
```

#### How to run the .venv python environment

run `source .venv/bin/activate` to activate the python environment.