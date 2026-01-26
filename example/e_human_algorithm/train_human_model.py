import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from example.e_human_algorithm.human_classifier import human_classify
from example.e_data.fetch_data import load_iris_data
from sklearn.model_selection import train_test_split

# This section of code separates the whole data-set into training and testing data.
df, target_name = load_iris_data()
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df[target_name]
)

# This section of code applies the human classification algorithm to the test data.
test_df['human_prediction'] = test_df['petal width'].apply(human_classify)
test_df['correct'] = test_df['human_prediction'] == test_df[target_name]
accuracy = (test_df['human_prediction'] == test_df[target_name]).mean()
print(f"Human classifier accuracy: {accuracy:.2%}")

# Here we print the confusion matrix to see how well the human classifier performed on the test-data subset.
conf_matrix = pd.crosstab(
    test_df[target_name],
    test_df['human_prediction'],
    rownames=['Actual'],
    colnames=['Predicted']
)
print(conf_matrix)

# Finally, we print one example of a failure case where the human classifier got the prediction wrong.
failure_row = test_df[test_df['human_prediction'] != test_df[target_name]].iloc[0]
print("\nFAILURE EXAMPLE")
print(failure_row[['sepal width', 'petal width', target_name, 'human_prediction']])



os.makedirs("example/e_ml_model/plots", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=test_df,
    x='sepal width',
    y='petal width',
    hue='correct',
    style='correct',
    s=100,
    palette={True: 'green', False: 'red'}
)

plt.title('Human Algorithm: Correct vs Incorrect Predictions')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Prediction Correct')
plt.grid(True)
plt.savefig('example/e_human_algorithm/plots/human_model_training_results.png', dpi=150)
plt.close()