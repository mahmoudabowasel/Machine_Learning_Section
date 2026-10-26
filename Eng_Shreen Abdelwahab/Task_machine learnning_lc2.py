# محمود كمال ابو واصل عوض
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.array([[2], [4], [6], [8], [10]])
y = np.array([40, 50, 65, 80, 90])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
residuals = y - predictions

mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

hours = 7
predicted_grade = model.predict([[hours]])[0]

print("Linear Regression")
print("-----------------")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Equation: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x")
print()
print("Actual and Predicted Grades")
print("Hours\tActual\tPredicted\tResidual")
for x_value, actual, predicted, residual in zip(X.flatten(), y, predictions, residuals):
    print(f"{x_value}\t{actual}\t{predicted:.2f}\t\t{residual:.2f}")

print()
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"R² Percentage: {r2 * 100:.2f}%")
print(f"Predicted grade for {hours} study hours: {predicted_grade:.2f}")

x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.scatter(X, y, label="Actual Grades")
plt.plot(x_line, y_line, label="Regression Line")
plt.scatter([hours], [predicted_grade], marker="x", s=100, label="Prediction for 7 Hours")
plt.xlabel("Study Hours")
plt.ylabel("Exam Grade")
plt.title("Linear Regression: Study Hours vs Exam Grade")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
