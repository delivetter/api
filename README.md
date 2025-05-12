# Delivetter – Simulation API

This repository contains the backend API for **Delivetter**, built with **FastAPI**. It exposes an endpoint to run delivery simulations, comparing the efficiency of an autonomous robot (**Ona**) and a traditional van with a driver.

---

## 🚀 Features

* REST API endpoint to trigger delivery simulations
* Accepts structured input parameters such as delivery locations and delivery method
* Returns cost and time estimates for each model
* Built with FastAPI for performance and automatic documentation

---

## 📆 Requirements

* Python ≥ 3.10
* `uvicorn`, `fastapi`, and other dependencies listed in `requirements.txt`

---

## 🧑‍💻 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/delivetter/api
cd api
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the server:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`