# AI Banking Customer Assistant

A machine learning-based banking customer assistant that predicts a user's banking intent from their message and returns a predefined response.

## Project Overview

The project treats customer support as an **intent classification** problem.

The workflow is:

1. Load the Banking77 dataset.
2. Explore and check the data.
3. Split the training data into training and validation sets.
4. Convert text into TF-IDF features using word unigrams and bigrams.
5. Train and compare:
   - Logistic Regression
   - Linear SVM
6. Evaluate the models using accuracy, precision, recall, and F1-score.
7. Select Linear SVM as the final model.
8. Evaluate the final model on the separate test set.
9. Save the trained TF-IDF vectorizer and SVM model.
10. Use Flask to connect the model to a simple web interface.

## Dataset

The project uses the **Banking77** dataset.

- Training samples: 10,003
- Test samples: 3,080
- Number of intents: 77
- Training columns: `text` and `category`

The training data was split into 80% training and 20% validation using stratification.

## Machine Learning Approach

### TF-IDF

`TfidfVectorizer` was used with:

```python
ngram_range=(1, 2)
```

This represents the text using unigrams and bigrams.

### Models

Two models were compared:

- Logistic Regression
- Linear SVM (`LinearSVC`)

Linear SVM performed better on the validation set and was used as the final model.

## Results

### Validation Set

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Logistic Regression | 84.31% | 84.22% |
| Linear SVM | **88.31%** | **88.35%** |

For the Linear SVM, the validation results were:

- Accuracy: **88.31%**
- Weighted Precision: **88.79%**
- Weighted Recall: **88.31%**
- Weighted F1-score: **88.35%**
- Macro F1-score: **88.41%**

### Test Set

The final Linear SVM achieved:

- Test Accuracy: **88.34%**
- Test Weighted F1-score: **88.36%**

## Web Application

The project includes a Flask application.

The application:

1. Receives a customer message.
2. Transforms the message using the saved TF-IDF vectorizer.
3. Predicts the banking intent using the saved Linear SVM model.
4. Returns the predicted intent and a predefined response.

## Project Structure

```text
AI_Banking_Customer_Assistant/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── templates/
│   └── index.html
│
├── AI_Banking_Customer_Assistant.ipynb
├── app.py
├── svm_model.pkl
├── tfidf.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Install the requirements

```bash
pip install -r requirements.txt
```

### 2. Make sure these files are in the project folder

```text
svm_model.pkl
tfidf.pkl
templates/index.html
```

The Flask app loads the saved model and TF-IDF vectorizer from the project directory.

### 3. Run the Flask application

```bash
python app.py
```

Then open the local address shown by Flask in your browser.

## Notes

This is a machine learning customer-support prototype. The responses are predefined and the system does not connect to a real bank account or perform banking transactions.

The model is designed to classify the intent of the customer's message rather than generate free-form answers.
