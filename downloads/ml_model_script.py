
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

# Load Data
data = pd.read_csv('Google_Drive_Data.csv')

# Plotting
plt.figure()
sns.scatterplot(data=data, x='age', y='age')
plt.show()
plt.close()



# Preprocessing
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X.select_dtypes(include='number'))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2)

# Model Training
model = {
    'logistic_regression': LogisticRegression(),
    'linear_regression': LinearRegression(),
    'svm': SVC(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier()
}['logistic_regression']

model.fit(X_train, y_train)


# Evaluation
predictions = model.predict(X_test)


print("Model Performance (Classification):")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, predictions, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, predictions, average='weighted'):.4f}")


