# Tourism Package Prediction

MLOps pipeline for "Visit with Us" that predicts whether a customer will
buy the new Wellness Tourism Package.

**Workflow:** GitHub + GitHub Actions + Streamlit Community Cloud (the
course office confirmed this is an acceptable alternative to the Hugging
Face-based rubric example, since HF now needs paid tiers for some
features). The GitHub repo holds the data and the trained model, GitHub
Actions runs the pipeline and commits the model back to `main`, and
Streamlit Community Cloud serves the app straight from the repo. No
Hugging Face account and no GitHub secrets are needed.

## Structure

```
tourism_project/
  data/tourism.csv              raw dataset
  model_building/
    data_register.py            checks the dataset
    prep.py                     cleans data, splits train/test
    train.py                    trains + tunes models, logs to MLflow, saves the best one
    requirements.txt
  deployment/
    app.py                      Streamlit app
    requirements.txt
    runtime.txt
    best_tourism_package_model_v1.joblib   trained model (committed by the pipeline)
.github/workflows/pipeline.yml  CI/CD pipeline
requirements.txt
Learner_Template_Notebook_AML_and_MLOps_Project.ipynb   solution notebook
```

## Pipeline

Three jobs run in order on every push to `main`:

1. `register-dataset` — checks the dataset looks right
2. `data-prep` — cleans the data, splits it into train/test, and commits
   the splits back to `main`
3. `model-training` — trains a Decision Tree, Random Forest, and XGBoost,
   tunes each with GridSearchCV, logs everything to MLflow, picks the best
   by F1 score, and commits the model file back to `main`

Streamlit Community Cloud watches the repo, so once the model is committed
the app redeploys automatically.

## Setup

1. Push this repo to GitHub (see the notebook's "GitHub Authentication and
   Push Files" section — needs a GitHub token just for this one push).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, and point it at
   `tourism_project/deployment/app.py` on the `main` branch.

That's it — pushing to `main` from then on redeploys the app automatically.

## Running locally

Root `requirements.txt` is the deployment-only set (what Streamlit Cloud
actually installs — keep it minimal). For training/EDA, install the
model_building one too:

```bash
pip install -r requirements.txt
pip install -r tourism_project/model_building/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
streamlit run tourism_project/deployment/app.py
```

## Output Evaluation

- GitHub repo: `https://github.com/tysoowealthmanagement-arch/Visit-with-Us-MLOPS-Project`
- Streamlit app: `https://visit-with-us-prediction-system.streamlit.app/`

