from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def evaluate_model(pipeline, X_train, y_train, X_test, y_test, X, y):
    print("\n🔍 Cross-Validation with 10 Folds (MAE):")
    kfold_10 = KFold(n_splits=10, shuffle=True, random_state=42)

    cv_mae_train = cross_val_score(pipeline, X_train, y_train, cv=kfold_10, scoring='neg_mean_absolute_error')
    print("Mean Absolute Error (Train):", -np.mean(cv_mae_train))

    cv_mae_test = cross_val_score(pipeline, X_test, y_test, cv=kfold_10, scoring='neg_mean_absolute_error')
    print("Mean Absolute Error (Test):", -np.mean(cv_mae_test))

    cv_mae_all = cross_val_score(pipeline, X, y, cv=kfold_10, scoring='neg_mean_absolute_error')
    print("Mean Absolute Error (All Data):", -np.mean(cv_mae_all))

    print("\n📈 Cross-Validation with 25 Folds (R2 Score):")
    kfold_25 = KFold(n_splits=25, shuffle=True, random_state=42)

    cv_r2_train = cross_val_score(pipeline, X_train, y_train, cv=kfold_25, scoring='r2')
    print("R2 Score (Train):", np.mean(cv_r2_train))

    cv_r2_test = cross_val_score(pipeline, X_test, y_test, cv=kfold_25, scoring='r2')
    print("R2 Score (Test):", np.mean(cv_r2_test))
