import optuna
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

def optimize_model(X_train, y_train, X_test, y_test, preprocessor, n_trials=500):
    def objective(trial):
        model = XGBRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 500),
            max_depth=trial.suggest_int('max_depth', 3, 15),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
            gamma=trial.suggest_float('gamma', 0.0, 5.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 5.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
            tree_method='hist',
            device='cuda',
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return r2_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    final_model = XGBRegressor(
        **best_params,
        tree_method='hist',
        device='cuda',
        random_state=42
    )
    final_model.fit(X_train, y_train)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', final_model)
    ])
    return pipeline, best_params
