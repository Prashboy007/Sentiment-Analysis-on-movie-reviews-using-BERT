# Sentiment Analysis on Movie Reviews using BERT

## Overview

This project aims to perform sentiment analysis on movie reviews using BERT (Bidirectional Encoder Representations from Transformers). Sentiment analysis is a technique used to determine whether a piece of text expresses a positive, negative, or neutral sentiment. By leveraging the power of BERT, this project enhances the accuracy of sentiment classification.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project is the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). This dataset contains movie reviews from IMDB and includes a balance of positive and negative reviews.

### Features:

- **text**: The review text.
- **sentiment**: The sentiment label (positive or negative).

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-bert.git
    cd sentiment-analysis-bert
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project directory structure is as follows:

sentiment-analysis-bert/ │ ├── data/
│ ├── imdb_reviews.csv # Original dataset │ ├── notebooks/
│ ├── Data_Preprocessing.ipynb # Jupyter notebook for data preprocessing │ ├── Model_Training.ipynb # Jupyter notebook for model training and evaluation │ ├── models/
│ ├── bert_model.pth # Trained BERT model │ ├── src/
│ ├── preprocess.py # Script for data preprocessing │ ├── train.py # Script for model training │ └── evaluate.py # Script for model evaluation │ ├── README.md # Project README file ├── requirements.txt # Python dependencies └── LICENSE # License file


## Data Preprocessing

The data preprocessing steps include:

1. **Loading Data**: Read the dataset from CSV and convert it into a suitable format for BERT.
2. **Tokenization**: Use BERT’s tokenizer to convert the text into tokens.
3. **Padding and Truncation**: Ensure that all text sequences are of the same length.
4. **Encoding**: Convert tokens into BERT input IDs and create attention masks.

## Modeling

This project uses BERT for sentiment analysis, and the steps include:

1. **Model Selection**: Use `bert-base-uncased` from Hugging Face's Transformers library.
2. **Fine-tuning**: Train the BERT model on the preprocessed movie reviews dataset.
3. **Optimization**: Use appropriate hyperparameters for fine-tuning, such as learning rate and batch size.

## Evaluation

The model's performance is evaluated based on:

- **Accuracy**: The proportion of correctly classified reviews.
- **Precision**: The number of true positives divided by the sum of true and false positives.
- **Recall**: The number of true positives divided by the sum of true positives and false negatives.
- **F1-Score**: The harmonic mean of precision and recall.

## Results

The fine-tuned BERT model achieved the following results on the test set:

- **Accuracy**: 89%
- **Precision**: 90.0%
- **Recall**: 89.0%
- **F1-Score**: 89.0%

## Contributing

Contributions are welcome! If you find any issues or want to contribute to the project, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
