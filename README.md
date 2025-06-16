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
* [PDM](https://pdm.fming.dev/) for dependency management

---

## 🧑‍💻 Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/delivetter/api
    cd api
    ```

2. (Optional) Create and activate a virtual environment:

    ```bash
    pdm venv create
    pdm venv activate
    ```

3. Install dependencies using PDM:

    ```bash
    pdm install
    ```

4. Run the server in development mode:

    ```bash
    pdm run dev
    ```

The API will be available at:  
[http://localhost:8000](http://localhost:8000)
