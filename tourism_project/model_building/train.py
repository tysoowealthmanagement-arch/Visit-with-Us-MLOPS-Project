# Model Training
# Train a few models, tune them, track everything with MLflow, and save the
# best one for the Streamlit app to use.

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

Xtrain = pd.read_csv("tourism_project/data/Xtrain.csv")
Xtest = pd.read_csv("tourism_project/data/Xtest.csv")
ytrain = pd.read_csv("tourism_project/data/ytrain.csv").iloc[:, 0]
ytest = pd.read_csv("tourism_project/data/ytest.csv").iloc[:, 0]

numeric_features = [
    "Age", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "PitchSatisfactionScore",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "CityTier", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Passport", "OwnCar", "Designation",
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# the target is imbalanced (~19% positive), seen during EDA, so we
# weight the minority class instead of leaving the models to ignore it
neg, pos = (ytrain == 0).sum(), (ytrain == 1).sum()
scale_pos_weight = neg / pos

models = {
    "decision_tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {"classifier__max_depth": [3, 5, 7]},
    ),
    "random_forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {"classifier__n_estimators": [100, 200]},
    ),
    "xgboost": (
        XGBClassifier(random_state=42, eval_metric="logloss", scale_pos_weight=scale_pos_weight),
        {"classifier__n_estimators": [100, 200], "classifier__max_depth": [3, 5]},
    ),
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("tourism-package-prediction")

best_model = None
best_f1 = 0
best_name = ""

for name, (clf, param_grid) in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")

    with mlflow.start_run(run_name=name):
        grid.fit(Xtrain, ytrain)
        preds = grid.predict(Xtest)

        acc = accuracy_score(ytest, preds)
        f1 = f1_score(ytest, preds)
        precision = precision_score(ytest, preds)
        recall = recall_score(ytest, preds)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(grid.best_estimator_, name="model", serialization_format="cloudpickle")

        print(f"{name}: accuracy={acc:.3f} f1={f1:.3f} precision={precision:.3f} recall={recall:.3f}")

        # pick the best model by F1 score, since we care more about
        # correctly catching buyers than plain accuracy on this imbalanced data
        if f1 > best_f1:
            best_f1 = f1
            best_model = grid.best_estimator_
            best_name = name

print(f"\nBest model: {best_name} (F1={best_f1:.3f})")

joblib.dump(best_model, "tourism_project/deployment/best_tourism_package_model_v1.joblib")
