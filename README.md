# DDoS Attack Detection using Autoencoders

## Project Overview

This project aims to detect Distributed Denial of Service (DDoS) attacks by analyzing network traffic data using autoencoders. The project utilizes the CIC-DDoS2019 dataset from Kaggle, which contains labeled network traffic data with various types of DDoS attacks.

## Dataset

The CIC-DDoS2019 dataset is used for training and evaluating the model. It contains a variety of features extracted from network traffic, including source and destination IP addresses, port numbers, packet lengths, and flow durations.

## Methodology

1. **Data Preprocessing:** The dataset is preprocessed to handle missing values, remove duplicates, encode categorical features, and select relevant features based on correlation with the target variable (Label).

2. **Autoencoder Model:** An autoencoder, an unsupervised learning model, is used for anomaly detection. The autoencoder is trained on normal network traffic data to learn the patterns of benign behavior.

3. **Anomaly Detection:** A threshold is determined based on the reconstruction error of the autoencoder on the training data. Instances with a reconstruction error exceeding the threshold are classified as anomalies (potential DDoS attacks).

## Evaluation

The model's performance is evaluated using the following metrics:

*   **Mean Squared Error (MSE):** Measures the average squared difference between the original and reconstructed data points.
*   **R-squared (R2):** Represents the proportion of variance in the dependent variable that is explained by the independent variables.
*   **Classification Report:** Provides precision, recall, F1-score, and support for each class (normal and anomaly).

## Results

The model achieved a high R-squared value and low MSE, indicating good reconstruction performance. The classification report showed promising results in terms of precision, recall, and F1-score, demonstrating the model's ability to effectively detect DDoS attacks.

## Dependencies

*   Python 3.x
*   Pandas
*   NumPy
*   Scikit-learn
*   TensorFlow
*   Keras
*   Matplotlib
*   Seaborn
*   Kaggle API

## Usage

1. **Install dependencies:** `pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn kaggle`
2. **Download the CIC-DDoS2019 dataset from Kaggle.**
3. **Run the Jupyter Notebook:** Execute the code cells in the notebook to train the model and evaluate its performance.

## Conclusion

This project demonstrates the effectiveness of using autoencoders for DDoS attack detection. The model achieved promising results in identifying anomalies in network traffic data. Further research can explore different autoencoder architectures and hyperparameters to improve the model's performance.


## License

[MIT License](https://opensource.org/licenses/MIT)
