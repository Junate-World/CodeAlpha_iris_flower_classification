# Iris Flower Classification

This project involves building a classification model to predict the species of an Iris flower based on its features, including:
- Sepal length
- Sepal width
- Petal length
- Petal width

The dataset used is the famous Iris dataset, which contains 150 samples of three species of Iris flowers: **Iris-setosa**, **Iris-versicolor**, and **Iris-virginica**.

## Project Features

### 1. Data Preprocessing
- Removed unnecessary columns (e.g., `Id`).
- Checked and handled missing values (none were found).
- Conducted exploratory data analysis (EDA) including visualizations of feature distributions and correlations.

### 2. Feature Selection
- Conducted correlation analysis using a heatmap.
- Observed strong positive correlation between `PetalLengthCm` and `PetalWidthCm`.
- Decided to retain all features for training.

### 3. Model Training
- Split the dataset into training (80%) and testing (20%) sets.
- Trained a **Random Forest Classifier** for classification.
- Achieved 100% accuracy on the test set.

### 4. Model Evaluation
- Evaluated the model using metrics:
  - **Accuracy**: 1.0
  - **Classification Report**: Precision, Recall, and F1-score were perfect for all classes.
  - **Confusion Matrix**: The model correctly classified all test samples.
- Compared predicted labels with actual labels.

## Requirements

Install the necessary libraries using the following command:

```bash
pip install -r requirements.txt
```

### Libraries Used
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## Running the Project

1. Clone this repository:

```bash
git clone <repository_url>
```

2. Navigate to the project directory:

```bash
cd Iris Flower Classification
```

3. Run the script:

```bash
python CodeAlpha_iris_flower_classification.py
```

## Results

The model demonstrated excellent performance, achieving perfect accuracy on the test set. The classification results show the capability of Random Forest to handle this relatively small dataset effectively.

## Project Files
- `Iris.csv`: Dataset used for training and testing.
- `CodeAlpha_iris_flower_classification.py`: Python script containing data preprocessing, model training, and evaluation.
- `README.md`: Documentation for the project.
- `.gitignore`: Files to be ignored by version control.

## License

This project is licensed under the MIT License. See `LICENSE` for more information.
#   C o d e A l p h a _ i r i s _ f l o w e r _ c l a s s i f i c a t i o n  
 