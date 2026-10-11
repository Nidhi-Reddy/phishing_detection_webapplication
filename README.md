# Phishing Detection Web Application
## Overview

The **Phishing Detection Web Application** is a machine learning-based system designed to identify potentially malicious phishing URLs and distinguish them from legitimate URLs. It extracts relevant URL-based features and uses a trained classification model to predict whether a URL is legitimate or suspicious.
The application provides a simple web interface for submitting URLs and viewing prediction results. It also includes model evaluation and visualization components to support the analysis of classification performance.

## Objectives

- Detect potentially fraudulent and phishing URLs using machine learning.
- Extract relevant lexical and structural features from submitted URLs.
- Train and evaluate classification models for phishing detection.
- Compare machine learning models and optimization techniques.
- Provide an accessible web interface for URL classification.
- Store prediction history for reference and analysis.

## Features

- **URL Feature Extraction:** Extracts features such as URL length, IP address usage, suspicious URL patterns, domain characteristics, and HTTPS usage.
- **Machine Learning Classification:** Uses a trained classification model to predict whether a URL is phishing or legitimate.
- **Model Evaluation:** Supports analysis using classification metrics and performance visualizations.
- **Performance Comparison:** Includes result files and plots for comparing machine learning models and optimization techniques.
- **Web Interface:** Provides a frontend for interacting with the detection system.
- **Prediction History:** Uses SQLite to store prediction records, depending on the application configuration.

## Technologies Used

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Backend Framework | Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| URL Processing | Python `re`, `urllib.parse` |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Visualization | Matplotlib, Seaborn |

## Project Structure

```text
phishing_detection_webapplication/
├── backend/
│   ├── app.py
│   ├── feature_extraction.py
│   ├── deployment_training.py
│   ├── deployment_model.pkl
│   ├── feature_columns.pkl
│   ├── model_training.py
│   ├── advanced_evaluation_plots.py
│   ├── results_visualization.py
│   ├── experiment_results.csv
│   ├── model_results.csv
│   ├── requirements.txt
│   └── ...
├── database/
│   └── predictions.db
├── frontend/
│   ├── static/
│   │   └── style.css
│   └── templates/
├── .gitignore
└── README.md
```

*Note: The structure above represents the main files and folders. Some filenames or locations may differ depending on the final project configuration.*

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- pip package manager
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/Nidhi-Reddy/phishing_detection_webapplication.git
cd phishing_detection_webapplication
```

### 2. Install Dependencies

If `requirements.txt` is located inside the `backend` directory, run:

```bash
cd backend
python -m pip install -r requirements.txt
```

If a requirements file is located elsewhere, use its corresponding path.

### 3. Start the Application

From the `backend` directory, run:

```bash
python app.py
```

If the application starts successfully, open the local address displayed in the terminal, commonly:

```text
http://127.0.0.1:5000
```

The exact startup command and URL depend on the Flask application configuration.

## How to Use

1. Start the Flask application.
2. Open the web application in a browser.
3. Enter the URL you want to analyze.
4. Submit the URL for prediction.
5. View the classification result returned by the trained model.
6. Review saved prediction records if history functionality is enabled.

## Model Evaluation

The project includes scripts and result files for evaluating classification models and visualizing their performance.

Evaluation can include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve and AUC
- Precision-recall curve

The repository also contains visualization files for model performance and optimization comparisons. Actual evaluation results should be interpreted using the generated metrics and the corresponding test dataset.

## Security and Limitations

- The application performs URL-based classification and does not guarantee that every phishing website will be detected.
- Model predictions depend on the training dataset, extracted features, and classification threshold.
- HTTPS alone does not establish that a website is trustworthy.
- The application should not automatically visit or open submitted URLs unless that behavior is explicitly implemented and secured.
- Prediction databases and configuration files should be reviewed before public release to prevent accidental exposure of sensitive information.

## Future Enhancements

- Expand feature extraction using additional URL and domain characteristics.
- Improve detection performance through systematic model evaluation and hyperparameter tuning.
- Integrate updated phishing datasets.
- Enhance the user interface and prediction history dashboard.
- Add monitoring and periodic model evaluation.

## Author

**Srinidhi Reddy**

GitHub: [Nidhi-Reddy](https://github.com/Nidhi-Reddy)

## Disclaimer

This project is developed for educational and research purposes. Its predictions should be treated as an additional security signal rather than a guarantee that a website is safe or malicious.
