# spam-email-detection

# Problem statement 
The goal of this project is to develop a machine learning model capable of accurately classifying emails as either spam or not spam (ham) based on their content. The project will leverage labeled email datasets for training and evaluation. The process includes handling the entire machine learning pipeline: text preprocessing, feature extraction using scikit-learn's CountVectorizer, model training with the Multinomial Naive Bayes algorithm, hyperparameter tuning, and performance evaluation. The final model will be saved and deployed to serve as the backend for an automated email spam detection system.

# Project file and Input data


# Process

**1.Data Preparation**

**Steps:**

Loaded the dataset using the .read_csv function in Jupyter Notebook.

Analyzed the distribution of spam and non-spam emails using the groupby function.

Added a binary column, spam, assigning 1 for spam emails and 0 for non-spam emails.

**2.Data Splitting**

**Steps:**

Used train_test_split from scikit-learn to divide the dataset into training and testing subsets.

Set X as the Message column (input) and Y as the spam column (target value).

Text Vectorization

Converted the Message column (text data) into numerical vectors using scikit-learn's CountVectorizer, enabling compatibility with machine learning models.

**3.Model Selection and Training**

**Steps:**

Chose the Multinomial Naive Bayes algorithm for its effectiveness in classifying discrete data.

Trained the model on the training data and evaluated its performance on the test data.

**4.Model Evaluation**
Tested the model on random samples, accurately classifying emails as spam or non-spam.

Achieved an impressive accuracy of 98% on the test data.

Compared with other classifiers like Random Forest and Support Vector Machines (SVM), which underperformed in this classification task.

**Results:**
Finalized Multinomial Naive Bayes as the best model for this project due to its high accuracy and reliable performance.

# Outcome
The Multinomial Naive Bayes model achieved 98% accuracy on the test data, proving to be the most effective solution for classifying emails as spam or non-spam. The model is ready for integration into systems requiring automated email spam classification.

**Model Saving:**
Saved the trained model for deployment in future applications, enabling real-time email spam detection.
