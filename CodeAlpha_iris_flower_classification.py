import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

#load the dataset
df = pd.read_csv("Iris.csv")
print(df.head())
print(df.info())  # Overview of data types and non-null values
print(df.describe())  # Summary statistics for numerical columns
print(df.isnull().sum())  # Check for missing values

#Viewing the species unique values before transformation
#print(df["Species"].unique())

# Drop the 'Id' column
df.drop(columns=["Id"], inplace=True)

# Encode the target variable
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

# Display the transformed dataset
#print(df.head())

#Viewing the species unique values after transformation
#print(df["Species"].unique())

# Pair plot to observe feature distribution and class separation
#sns.pairplot(df, hue="Species")
#plt.show()

# Split the dataset into training and testing sets
X = df.drop(columns=["Species"])  # Features
y = df["Species"]  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
#print("Model Training Successfully completed")

# Predict on the test set
y_pred = model.predict(X_test)


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Feature importance
importances = model.feature_importances_
feature_names = X.columns

# Plot feature importance
#sns.barplot(x=importances, y=feature_names)
#plt.title("Feature Importance")
#plt.grid()
#plt.show()

# Drop non-numeric columns
numeric_features = df.drop(columns=["Species"])

# Calculate correlation matrix
correlation_matrix = numeric_features.corr()

# Plot the heatmap
#plt.figure(figsize=(8, 6))
#sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
#plt.title("Correlation Heatmap")
#plt.show()