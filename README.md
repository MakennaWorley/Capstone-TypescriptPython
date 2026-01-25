## 🔬 Probabilistic Ancestral Inference: Proof of Concept (PoC)

This repository contains the Proof of Concept (PoC) for a project exploring the use of probabilistic models to reconstruct missing ancestral genotypes from incomplete genetic data.

The PoC focuses on a core deliverable: a working system capable of inferring masked ancestral genotypes in simulated **_Drosophila melanogaster_ (fruit fly) populations** using a **Bayesian inference model**.

---

### 🌟 Project Goal

The primary goal of the PoC is to establish the technical and scientific feasibility of the inference framework by demonstrating that a **Bayesian model** trained on partially masked simulation data can accurately recover missing genotypes and visualize the results through an interactive dashboard.

### 🛠️ Key Components

The PoC system is built upon the following technologies and methodologies:

* **Model Organism:** Simulated **_Drosophila melanogaster_** populations, which provide a well-characterized system for controlled modeline.
* **Simulation:** Multi-generational population data generated using the forward-time and coalescent-based simulation frameworks **simuPOP** and **msprime**.
* **Inference Model:** A baseline **Bayesian model** developed using **PyMC** (built on PyTensor) to infer missing ancestral genotypes.
* **Visualization:** A lightweight **Streamlit dashboard** for visualization, loading datasets, running inference, and viewing performance metrics.

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

### 🎯 Proof of Concept Deliverables
The initial phase includes the following specific tasks:

- Simulate multi-generational fruit-fly populations with known inheritance rules using simuPOP and msprime.
- Develop and train a baseline Bayesian model to infer missing ancestral genotypes.
- Generate three independent datasets (training, validation, and testing) to ensure statistical robustness.
- Implement a lightweight Streamlit dashboard for visualization and performance monitoring.
- Evaluate inference accuracy using chi-square and likelihood-based metrics.

### 📝 Evaluation
Evaluation for the PoC focuses on Accuracy and Usability.


- Accuracy: Comparing predicted vs. known genotypes using statistical metrics (precision, recall, chi-square).
- Usability: Ensuring the dashboard clearly communicates results and model uncertainty.

The PoC serves as the Absolute Minimum deliverable for the project.