import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
import optuna
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report
)
import joblib

def load_features():
    X_train = pd.read_csv("data/processed/X_train_fe.csv")
    X_test  = pd.read_csv("data/processed/X_test_fe.csv")
    y_train = pd.read_csv("data/processed/y_train_fe.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test_fe.csv").squeeze()
    print("Features loaded!")
    print("X_train:", X_train.shape)
    print("X_test: ", X_test.shape)
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auc       = roc_auc_score(y_test, y_pred_prob)
    f1        = f1_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {"auc": auc, "f1": f1, "recall": recall, "precision": precision}

def train_baseline(X_train, X_test, y_train, y_test):
    # set mlflow experiment
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="baseline_xgboost"):
        params = {
            "n_estimators":     100,
            "max_depth":        5,
            "learning_rate":    0.1,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric":      "logloss",
            "random_state":     42
        }

        # train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # log params and metrics to mlflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # log model artifact
        mlflow.xgboost.log_model(model, "model")

        # save locally too
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/baseline_model.pkl")
        print("\nBaseline model saved!")

        return model, metrics

def objective(trial, X_train, X_test, y_train, y_test):
    # optuna suggests hyperparameters automatically
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "random_state":     42
    }

    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}"):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

    return metrics["auc"]

def tune_model(X_train, X_test, y_train, y_test, n_trials=10):
    print("\nStarting hyperparameter tuning with Optuna...")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test),
        n_trials=n_trials
    )

    print(f"\nBest trial AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params

def train_best_model(X_train, X_test, y_train, y_test, best_params):
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="best_model"):
        best_params.update({
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42
        })

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train, verbose=False)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "best_model")

        # save best model locally
        joblib.dump(model, "models/best_model.pkl")
        print("\nBest model saved to models/best_model.pkl")

        return model, metrics

if __name__ == "__main__":
    # load features
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    
    X_train, X_test, y_train, y_test = load_features()

    # step 1 - train baseline
    print("\n--- Training Baseline Model ---")
    baseline_model, baseline_metrics = train_baseline(
        X_train, X_test, y_train, y_test
    )

    # step 2 - tune with optuna
    print("\n--- Tuning Hyperparameters ---")
    best_params = tune_model(
        X_train, X_test, y_train, y_test,
        n_trials=10
    )

    # step 3 - train best model
    print("\n--- Training Best Model ---")
    best_model, best_metrics = train_best_model(
        X_train, X_test, y_train, y_test, best_params
    )

    print("\n--- Final Results ---")
    print(f"Baseline AUC: {baseline_metrics['auc']:.4f}")
    print(f"Best AUC:     {best_metrics['auc']:.4f}")