import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

bike_uci = pd.read_csv("day.csv")
print("bike_uciset loaded successfully!")
bike_uci.head() 

#dropping unwanted columns only if they exist
cols_to_drop = ['instant', 'dteday', 'casual', 'registered']
if all(col in bike_uci.columns for col in cols_to_drop):
    bike_uci = bike_uci.drop(cols_to_drop, axis=1)

print("Columns after cleaning:", list(bike_uci.columns))

print(bike_uci.info())
print("\nSummary Statistics:")
print(bike_uci.describe())

print("\nMissing Values:\n", bike_uci.isnull().sum())

#visualising relationships
plt.figure(figsize=(10,5))
sns.scatterplot(x='temp', y='cnt', data=bike_uci)
plt.title('Temperature vs Bike Rentals', fontweight='bold')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='season', y='cnt', data=bike_uci)
plt.title('Season vs Bike Rentals', fontweight='bold')
plt.show() 

plt.figure(figsize=(10,5))
sns.boxplot(x='weathersit', y='cnt', data=bike_uci)
plt.title('Weather Situation vs Bike Rentals',fontweight='bold')
plt.show()

#correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

#computing correlation matrix
corr = bike_uci.corr(numeric_only=True)

#plotting correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontweight='bold')
plt.show()

#displaying top correlated features with bike count
print("\nTop correlated features with 'cnt':")
print(corr['cnt'].sort_values(ascending=False))

#preparing the data for modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#defining features (X) and target (y)
X = bike_uci.drop('cnt', axis=1)
y = bike_uci['cnt']

#splitting the bike_uci
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#normalising numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data prepared for modeling")
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

#training a random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#initialising model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)

#training
rf_model.fit(X_train_scaled, y_train)

#predicting
y_pred = rf_model.predict(X_test_scaled)

#evaluating
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Random Forest Performance:")
print(f"  Mean Squared Error: {mse:.2f}")
print(f"  R² Score: {r2:.3f}")

#feature importance
importances = rf_model.feature_importances_
feature_names = X.columns

#creating a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Feature Importance - Random Forest', fontweight='bold')
plt.show()

#trying a gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor

#initialising model
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)

#training
gb_model.fit(X_train_scaled, y_train)

#predicting
y_pred_gb = gb_model.predict(X_test_scaled)

#evaluating
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"Gradient Boosting Performance:")
print(f"  Mean Squared Error: {mse_gb:.2f}")
print(f"  R² Score: {r2_gb:.3f}")

#comparing and visualising random forest and gradient boosting
comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting'],
    'MSE': [mse, mse_gb],
    'R² Score': [r2, r2_gb]
})

plt.figure(figsize=(8,4))
sns.barplot(data=comparison, x='Model', y='R² Score', palette='crest')
plt.title('Model Performance Comparison (R² Score)')
plt.show()
comparison

#for fun: adding an interactive feature importance using plotly
import plotly.express as px

fig = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    title='Interactive Feature Importance (Random Forest)',
    color='Importance',
    color_continuous_scale='viridis'
)
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()

#hyperparameter tuning and optimization
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

#defining parameter grid
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2']
}

#initialising Randomized Search
rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='r2',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

#run tuning
rf_random.fit(X_train_scaled, y_train)

#best parameters
print("Best Parameters for Random Forest:")
print(rf_random.best_params_)

#evaluating optimized model
best_rf = rf_random.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)

mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"\nOptimized Random Forest Performance:")
print(f"  Mean Squared Error: {mse_best:.2f}")
print(f"  R² Score: {r2_best:.3f}") 

#gradient boosting hyperparameter tuning
#because GBR takes longer to train, a smaller search space will be used
from scipy.stats import uniform

param_dist_gb = {
    'n_estimators': randint(100, 400),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

gb_random = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=param_dist_gb,
    n_iter=20,
    scoring='r2',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

gb_random.fit(X_train_scaled, y_train)

print("Best Parameters for Gradient Boosting:")
print(gb_random.best_params_)

#evaluating optimized models
best_gb = gb_random.best_estimator_
y_pred_gb_best = best_gb.predict(X_test_scaled)

mse_gb_best = mean_squared_error(y_test, y_pred_gb_best)
r2_gb_best = r2_score(y_test, y_pred_gb_best)

print(f"\nOptimized Gradient Boosting Performance:")
print(f"  Mean Squared Error: {mse_gb_best:.2f}")
print(f"  R² Score: {r2_gb_best:.3f}")

#comparing optimized models
comparison_optimized = pd.DataFrame({
    'Model': ['Random Forest (Tuned)', 'Gradient Boosting (Tuned)'],
    'MSE': [mse_best, mse_gb_best],
    'R² Score': [r2_best, r2_gb_best]
})

plt.figure(figsize=(8,4))
sns.barplot(data=comparison_optimized, x='Model', y='R² Score', palette='mako')
plt.title('Tuned Model Performance Comparison (R² Score)')
plt.show()

comparison_optimized

#saving the best model used to deploy later
import joblib

joblib.dump(best_rf, 'best_random_forest_model.pkl')
joblib.dump(best_gb, 'best_gradient_boosting_model.pkl')

print("Models saved successfully!") 
