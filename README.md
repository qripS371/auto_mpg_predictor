# Auto MPG Predictor – Linear Regression CLI Tool

A Python CLI application that predicts a car's MPG (Miles Per Gallon) using multivariate linear regression. Trained on the Auto MPG dataset, the model uses features like displacement, horsepower, weight, acceleration, cylinders, and model year. Parameters are optimized via gradient descent for accurate and explainable predictions.

---

## Features

- Multivariate Linear Regression from scratch (no ML libraries)
- Parameter tuning using Gradient Descent
- Real-time training and prediction via CLI
- Handles missing data with preprocessing
- Educational baseline for understanding linear regression

---

## Installation

```bash
git clone https://github.com/qripS371/auto_mpg_predictor
cd auto_mpg_predictor
pip install -r requirements.txt
Usage
bash
Copy
Edit
python mpg_predictor.py
Make sure auto-mpg.csv is in the same directory as the script.

Requirements
Python 3.x

pandas

numpy

matplotlib

You can install all dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
Dataset
This project uses the classic Auto MPG dataset, which includes:

Displacement

Horsepower

Weight

Acceleration

Cylinders

Model Year

Make sure auto-mpg.csv is present in the root folder before executing the script.

Project Structure
cpp
Copy
Edit
.
├── mpg_predictor.py
├── auto-mpg.csv
├── requirements.txt
└── README.md
License
MIT License

Author
Prithvi
GitHub: qripS371

yaml
Copy
Edit

---

Let me know if you want a custom `requirements.txt` or a visual badge setup for GitHub too!
