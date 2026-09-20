import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/heart_disease.csv")

print("Dataset shape:", df.shape)
print(df.head())


# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------

# Target column in our dataset is "num"
X = df.drop("num", axis=1)
y = df["num"]


# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Create Pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        random_state=42
    ))
])


# --------------------------------------------------
# 5. Hyperparameter Grid
# --------------------------------------------------

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}


# --------------------------------------------------
# 6. Grid Search
# --------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


# --------------------------------------------------
# 7. MLflow Configuration
# --------------------------------------------------

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Heart Disease Hyperparameter Tuning"
)


# --------------------------------------------------
# 8. Run Hyperparameter Tuning
# --------------------------------------------------

with mlflow.start_run(
    run_name="Random Forest Hyperparameter Tuning"
):

    print("\nStarting hyperparameter tuning...")

    grid_search.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # 9. Best Parameters
    # --------------------------------------------------

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print("\nBest Parameters:")
    print(best_params)

    print("\nBest Cross-Validation Accuracy:")
    print(best_cv_score)


    # --------------------------------------------------
    # 10. Test Set Prediction
    # --------------------------------------------------

    predictions = best_model.predict(X_test)


    # --------------------------------------------------
    # 11. Calculate Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------
    # 12. Log Parameters
    # --------------------------------------------------

    mlflow.log_param(
        "model",
        "Random Forest"
    )

    mlflow.log_param(
        "cv",
        5
    )

    mlflow.log_param(
        "scoring",
        "accuracy"
    )

    for parameter, value in best_params.items():

        mlflow.log_param(
            parameter,
            value
        )


    # --------------------------------------------------
    # 13. Log Metrics
    # --------------------------------------------------

    mlflow.log_metric(
        "cv_accuracy",
        best_cv_score
    )

    mlflow.log_metric(
        "test_accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # --------------------------------------------------
    # 14. Log Best Model
    # --------------------------------------------------

    mlflow.sklearn.log_model(
        best_model,
        name="best_random_forest",
        serialization_format="cloudpickle",
        registered_model_name="HeartDiseaseModel",
    )


    # --------------------------------------------------
    # 15. Print Final Results
    # --------------------------------------------------

    print("\n===================================")
    print("BEST RANDOM FOREST MODEL")
    print("===================================")

    print("Best Parameters:")
    print(best_params)

    print("\nCV Accuracy:", best_cv_score)

    print("\nTest Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


print("\nHyperparameter tuning completed successfully.")