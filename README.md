````markdown
# CarePredict-AI: Client Satisfaction Prediction & Explanation

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
- [Deployment on Vercel](#deployment-on-vercel)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About the Project
CarePredict-AI is an intelligent application designed to predict and explain client satisfaction in clinical settings, specifically within an HIV clinic context. Leveraging machine learning and rule-based systems, it provides insights into factors influencing client satisfaction, offering actionable reasons and suggestions for quality improvement.

The application aims to empower clinical teams to proactively address areas of concern, enhance patient experience, and drive better care outcomes by understanding the 'why' behind satisfaction scores.

## Features
* **Client Satisfaction Prediction:** Predicts the satisfaction level (e.g., Very Dissatisfied, Dissatisfied, Neutral, Satisfied, Very Satisfied) based on various client and interaction data.
* **SHAP-based Feature Importance:** Utilizes SHAP (SHapley Additive exPlanations) values to identify the most influential factors driving a specific client's satisfaction prediction.
* **Rule-Based Explanations:** Incorporates predefined clinical rules to generate human-readable reasons and actionable suggestions for improving client satisfaction.
* **Generative AI Synthesis:** Integrates with a Generative AI model (OpenRouter) to synthesize SHAP insights and rule-based explanations into an executive summary, detailed analysis of drivers, and actionable recommendations.
* **Interactive Dashboard:** Provides a user-friendly web interface for inputting client data, triggering predictions, and visualizing explanations.
* **FastAPI Backend:** A robust and high-performance API serving predictions and explanations.
* **Static Web Pages:** Simple HTML/CSS frontend for the home page and dashboard.

## Technologies Used

**Backend:**
* Python 3.9+
* FastAPI: High-performance web framework for building APIs.
* scikit-learn: For the machine learning model (e.g., RandomForestClassifier).
* pandas: Data manipulation and analysis.
* numpy: Numerical operations.
* shap: SHapley Additive exPlanations for model interpretability.
* requests: For making HTTP requests to the Generative AI service.
* python-dotenv: For managing environment variables.

**Frontend:**
* HTML5
* CSS3
* JavaScript (for interacting with the API)

**Deployment:**
* Vercel: Serverless deployment platform.

**AI Service:**
* OpenRouter: For accessing various Generative AI models (e.g., mistralai/mistral-7b-instruct:free).

## Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Before you begin, ensure you have the following installed:
* Python 3.9 or higher
* `pip` (Python package installer)
* `venv` (Python virtual environment module, usually comes with Python)
* Git

### Local Development
1.  **Clone the repository:**
    ```bash
    git clone [git@github.com:presiZHai/CarePredict-AI.git](git@github.com:presiZHai/CarePredict-AI.git)
    cd CarePredict-AI
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Environment Variables:**
    Create a `.env` file in the root of your project directory (`CarePredict-AI/`) and add your OpenRouter API key:
    ```
    SATISFACTION_APP_KEY="your_openrouter_api_key_here"
    ```
    Replace `"your_openrouter_api_key_here"` with your actual API key.

5.  **Run the FastAPI application:**
    ```bash
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
    ```
    The application will start, and you should see output similar to this:
    ```
    INFO:     Will watch for changes in these directories: ['C:\\Users\\walea\\CarePredict-AI']
    INFO:     Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000) (Press CTRL+C to quit)
    INFO:     Started reloader process [...] using StatReload
    INFO:     All model artifacts loaded successfully for API.
    INFO:     Started server process [...]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```
6.  **Access the application:**
    * **Home Page:** Open your web browser and go to `http://127.0.0.1:8000/`
    * **Dashboard:** Navigate to `http://127.0.0.1:8000/dashboard`
    * **API Documentation (Swagger UI):** `http://127.0.0.1:8000/docs`
    * **API Documentation (ReDoc):** `http://127.0.0.1:8000/redoc`

### Deployment on Vercel
This project is configured for serverless deployment on Vercel.

1.  **Push to GitHub:** Ensure your project, including `requirements.txt` and `vercel.json` (as described in the Vercel deployment instructions), is pushed to a GitHub repository.
2.  **Vercel Account:** Create a free account on Vercel if you don't have one.
3.  **Import Project:** From your Vercel Dashboard, import your GitHub repository.
4.  **Configure Environment Variables:** Set the `SATISFACTION_APP_KEY` environment variable in your Vercel project settings.
5.  **Deploy:** Vercel will automatically detect the `vercel.json` and deploy your FastAPI application and static files.

Your deployed application will be available at a URL like `https://your-project-name.vercel.app`.

## API Endpoints
The FastAPI application exposes the following endpoints:

* `/` (GET): Serves the static home page (`index.html`).
* `/dashboard` (GET): Serves the static client satisfaction dashboard (`dashboard.html`).
* `/api/categories` (GET):
    * **Description:** Returns a list of categorical features and their unique values, used for populating dropdowns in the frontend.
    * **Response:** `{"column_name": ["value1", "value2", ...], ...}`
* `/api/predict_explain` (POST):
    * **Description:** Accepts client data, makes a satisfaction prediction, and generates comprehensive explanations (SHAP, rule-based, and AI-synthesized).
    * **Request Body (JSON Example):**
        ```json
        {
            "Age": 35,
            "Gender": "Female",
            "Employment_Grouped": "Employed",
            "Education_Grouped": "Higher Education",
            "Facility_Care_Dur_Years": 5,
            "HIV_Care_Duration_Ratio": 0.8,
            "Empathy_Listening_Interaction": 12,
            "Empathy_DecisionShare_Interaction": 13,
            "Exam_Explained": 3,
            "Discuss_NextSteps": 3
        }
        ```
    * **Response (JSON Example):**
        ```json
        {
            "prediction": "Satisfied",
            "confidence": "92.5%",
            "top_features": {
                "Empathy_Listening_Interaction": 0.52,
                "HIV_Care_Duration_Ratio": 0.31,
                "Facility_Care_Dur_Years": 0.25
            },
            "reasons": [
                "Strong empathy and active listening boosted client satisfaction."
            ],
            "suggestions": [
                "Encourage continued focus on empathetic listening."
            ],
            "genai_explanation": "### 1. Executive Summary\\nThis client is predicted to be 'Satisfied' with a high confidence of 92.5%. The primary drivers of this satisfaction are strong empathetic communication and active listening during interactions, along with a positive history in HIV care duration and overall facility care duration.",
            "shap_values": [0.1, -0.05, "..."],
            "shap_base_value": -0.8,
            "feature_values": [35, "Female", "..."],
            "feature_names": ["Age", "Gender", "..."]
        }
        ```

## Project Structure
````

CarePredict-AI/
├── app/
│   ├── **init**.py
│   ├── api.py                     \# Main FastAPI application
│   ├── explanation\_engine.py     \# Core logic for prediction, SHAP, rules, and GenAI
│   └── model\_utils.py            \# Model utilities
├── data/
│   └── processed_data.csv
├── model/                         \# Directory for your trained ML models
│   ├── categories.joblib           \# Serialized scikit-learn models and features 
│   ├── important_features.joblib         
│   ├── label_encoder.joblib
│   ├── top_categorical_features.joblib                 
│   └── top10_model.joblib 
├── static/
│   ├── index.html                 \# Home page
│   ├── dashboard.html             \# Interactive dashboard
│   ├── style.css                  \# Global stylesheets
│   └── images/                    \# Image assets
│       ├── ahfid.png
│       ├── customer-satisfaction.jpg
│       └── icons.jpg
├── .env                          \# Environment variables (for local development)
├── .gitignore                    \# Files/directories to ignore in Git
├── requirements.txt              \# Python dependencies
├── train_model.py
├── vercel.json                   \# Vercel deployment configuration
└── README.md                     \# This file

```

## Future Enhancements
* **Enhanced Data Validation:** Implement more robust input validation and error handling on the frontend and backend.
* **User Authentication:** Add user login/authentication to secure API endpoints.
* **Historical Data Analysis:** Incorporate functionality to analyze trends in satisfaction over time.
* **Database Integration:** Connect to a database to store client data and predictions for longitudinal analysis.
* **More Sophisticated Rules:** Expand the rule-based system with more granular and dynamic rules.
* **Improved UI/UX:** Enhance the dashboard's user interface and experience for better data visualization and interaction.
* **Model Retraining Pipeline:** Implement a process for periodically retraining and updating the ML model.

## Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Wale Ogundeji - abiodungndj@gmail.com

Project Link: [https://github.com/presiZHai/CarePredict-AI](https://github.com/presiZHai/CarePredict-AI)

Created with ❤️ by Wale Ogundeji
```