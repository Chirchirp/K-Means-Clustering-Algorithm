# Importing Libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import unittest

class KMeansApplication:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.model = None

    def load_data(self):
        # Load the dataset
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        if self.data.isnull().values.any():
            # Fill missing values with the mean of the column
            self.data.fillna(self.data.mean(), inplace=True)

        # Select numerical columns for scaling
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()

        # Scale the numerical features
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])

    def train_model(self, num_clusters):
        # Select features for clustering
        X = self.data[['Avg_Credit_Limit', 'Total_Credit_Cards']]

        # Train the K-Means model
        self.model = KMeans(n_clusters=num_clusters)
        self.model.fit(X)

    def get_clusters(self):
        # Return the cluster assignments
        return self.model.labels_

    def visualize_clusters(self):
        # Visualize the clusters
        if self.model is not None:
            X = self.data[['Avg_Credit_Limit', 'Total_Credit_Cards']]

            plt.figure(figsize=(8, 6))
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=self.model.labels_, cmap='viridis')
            plt.scatter(self.model.cluster_centers_[:, 0], self.model.cluster_centers_[:, 1], s=300, c='red',
                        label='Centroids')
            plt.title('K-Means Cluster Visualization')
            plt.xlabel('Avg_Credit_Limit')
            plt.ylabel('Total_Credit_Cards')
            plt.legend()
            plt.show()
        else:
            print("Model has not been trained. Please train the model before visualizing clusters.")

# Run the application
app = KMeansApplication('A:\\Machine Learning\\Credit_Card\\Credit_Card.csv')
app.load_data()
app.preprocess_data()
app.train_model(3)
app.visualize_clusters()

# Unit testing
class TestKMeansApplication(unittest.TestCase):
    def setUp(self):
        # Set up before each test method
        self.app = KMeansApplication('A:\\Machine Learning\\Credit_Card\\Credit_Card.csv')
        self.app.load_data()

    def test_preprocess_data(self):
        # Test the preprocess_data method
        original_data = self.app.data.copy()
        self.app.preprocess_data()
        self.assertFalse((original_data == self.app.data).any().any(), "Data should be different after preprocessing.")

    def test_train_model(self):
        # Test the train_model method
        self.app.preprocess_data()
        self.app.train_model(3)
        self.assertIsNotNone(self.app.model)

    def test_get_clusters(self):
        # Test the get_clusters method
        self.app.preprocess_data()
        self.app.train_model(3)
        clusters = self.app.get_clusters()
        self.assertEqual(len(clusters), len(self.app.data))

if __name__ == '__main__':
    # Run unit tests
    unittest.main()

    # Run the application
    app = KMeansApplication('A:\\Machine Learning\\Credit_Card\\Credit_Card.csv')
    app.load_data()
    app.preprocess_data()
    app.train_model(3)
    app.visualize_clusters()
