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

CarePredict-AI is an intelligent application designed to predict and explain client satisfaction in clinical settings, specifically within an HIV clinic context. It leverages machine learning and rule-based systems to offer insights into factors influencing satisfaction and gives actionable recommendations for quality improvement.

## Features

- **Client Satisfaction Prediction**: Predicts satisfaction levels using interaction and demographic data.
- **SHAP-based Feature Importance**: Highlights the most influential factors behind a prediction.
- **Rule-Based Explanations**: Uses predefined rules to generate human-readable reasons and suggestions.
- **Generative AI Synthesis**: Summarizes SHAP and rules into executive summaries and detailed recommendations.
- **Interactive Dashboard**: A simple web interface to input client data and view predictions.
- **FastAPI Backend**: High-performance API server.
- **Static Web Pages**: Home page and dashboard served using HTML/CSS.

## Technologies Used

**Backend**:
- Python 3.9+
- FastAPI
- scikit-learn
- pandas
- numpy
- shap
- requests
- python-dotenv

**Frontend**:
- HTML5
- CSS3
- JavaScript

**Deployment**:
- Vercel

**AI Service**:
- OpenRouter (e.g., mistralai/mistral-7b-instruct)

## Getting Started

Follow the steps below to get the project running locally.

### Prerequisites

Ensure the following are installed:
- Python 3.9+
- `pip`
- `venv`
- Git

### Local Development

1. Clone the repository:

    ```bash
    git clone git@github.com:presiZHai/CarePredict-AI.git
    cd CarePredict-AI
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up your `.env` file:

    ```env
    SATISFACTION_APP_KEY="your_openrouter_api_key_here"
    ```

5. Run the app:

    ```bash
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
    ```

6. Open in your browser:
- Home: `http://127.0.0.1:8000/`
- Dashboard: `http://127.0.0.1:8000/dashboard`
- Swagger Docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Deployment on Vercel

1. Push your project to GitHub.
2. Create a Vercel account and import your repo.
3. Add `SATISFACTION_APP_KEY` to Vercel environment variables.
4. Deploy. Vercel will detect your config from `vercel.json`.

## API Endpoints

- `/`: Serves the home page.
- `/dashboard`: Serves the dashboard.
- `/api/categories` (GET): Returns available options for categorical input fields.
- `/api/predict_explain` (POST): Takes client input and returns prediction, SHAP results, rule-based suggestions, and GenAI explanations.

Example request:

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

Example response:

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
    "genai_explanation": "### 1. Executive Summary\nThis client is predicted to be 'Satisfied' with a high confidence of 92.5%. The primary drivers of this satisfaction are strong empathetic communication and active listening during interactions, along with a positive history in HIV care duration and overall facility care duration.",
    "shap_values": [0.1, -0.05, "..."],
    "shap_base_value": -0.8,
    "feature_values": [35, "Female", "..."],
    "feature_names": ["Age", "Gender", "..."]
}
```

## Project Structure

```
CarePredict-AI/
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── explanation_engine.py
│   └── model_utils.py
├── data/
│   └── processed_data.csv
├── model/
│   ├── categories.joblib
│   ├── important_features.joblib
│   ├── label_encoder.joblib
│   ├── top_categorical_features.joblib
│   └── top10_model.joblib
├── static/
│   ├── index.html
│   ├── dashboard.html
│   ├── style.css
│   └── images/
│       ├── ahfid.png
│       ├── customer-satisfaction.jpg
│       └── icons.jpg
├── .env
├── .gitignore
├── requirements.txt
├── train_model.py
├── vercel.json
└── README.md
```

## Future Enhancements

- Add stronger input validation
- Secure the API with authentication
- Analyze trends with historical data
- Integrate with a database
- Expand the rule engine
- Improve UI and data viz
- Automate retraining and model updates

## Contributing

Pull requests are welcome.

1. Fork the repo
2. Create a new branch: `git checkout -b feature/YourFeature`
3. Commit: `git commit -m 'Add YourFeature'`
4. Push: `git push origin feature/YourFeature`
5. Open a PR

## License

MIT License. See `LICENSE`.

## Contact

**Wale Ogundeji**  
📧 abiodungndj@gmail.com  
🔗 [GitHub](https://github.com/presiZHai/CarePredict-AI)

Built with ❤️ by Wale Ogundeji