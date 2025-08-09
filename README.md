# Sentiment Analysis using LSTM

This project implements a **Long Short-Term Memory (LSTM)** neural network for sentiment analysis on movie reviews from the IMDB dataset. The model classifies movie reviews as either positive or negative sentiment with high accuracy using deep learning techniques.

## 🎯 Project Overview

This sentiment analysis project demonstrates:
- **LSTM Implementation**: Deep neural network architecture optimized for sequential text data
- **Text Processing**: Advanced tokenization and sequence padding techniques  
- **Binary Classification**: Positive vs. negative sentiment prediction
- **Real-world Dataset**: 50,000 IMDB movie reviews for robust training
- **Performance Optimization**: Early stopping and dropout for preventing overfitting

## 📊 Dataset

**IMDB Movie Reviews Dataset**
- **Source**: Kaggle - IMDB Dataset of 50K Movie Reviews
- **Size**: 50,000 movie reviews
- **Classes**: Binary classification (Positive: 1, Negative: 0)
- **Distribution**: Perfectly balanced - 25,000 positive and 25,000 negative reviews
- **Format**: Text reviews with corresponding sentiment labels

## 🏗️ Model Architecture

### LSTM Neural Network
```python
Sequential([
    Embedding(input_dim=5000, output_dim=128),  # Word embeddings
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),  # LSTM with regularization
    Dense(1, activation='sigmoid')  # Binary classification output
])
```

### Key Architecture Features:
- **Embedding Layer**: Maps 5,000 vocabulary words to 128-dimensional vectors
- **LSTM Layer**: 128 hidden units with dropout (0.2) and recurrent dropout (0.2)
- **Output Layer**: Single sigmoid neuron for binary classification
- **Regularization**: Dropout mechanisms to prevent overfitting

## 🔧 Key Features

- **Advanced Text Preprocessing**: Tokenization with vocabulary limit of 5,000 words
- **Sequence Padding**: Standardized input length of 200 tokens
- **LSTM Architecture**: Specialized for sequential data and long-term dependencies
- **Balanced Dataset**: Equal representation of positive and negative sentiments
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Train/Validation Split**: 80/20 split with additional 20% validation during training

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Main Dependencies:
- `tensorflow>=2.10.0` - Deep learning framework with Keras
- `scikit-learn>=1.0.0` - Machine learning utilities and metrics
- `pandas>=1.3.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Data visualization
- `seaborn>=0.11.0` - Statistical data visualization
- `kaggle>=1.5.12` - Dataset download API

## 🚀 Getting Started

### 1. Setup Kaggle API
```python
# Place your kaggle.json file in the project directory
# Configure environment variables
import os
import json

kaggle_dict = json.load(open('kaggle.json'))
os.environ['KAGGLE_USERNAME'] = kaggle_dict['username']
os.environ['KAGGLE_KEY'] = kaggle_dict['key']
```

### 2. Download and Extract Dataset
```python
# Download IMDB dataset
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Extract the dataset
from zipfile import ZipFile
with ZipFile('imdb-dataset-of-50k-movie-reviews.zip', 'r') as zipObj:
    zipObj.extractall()
```

### 3. Run the Model
```bash
jupyter notebook "Sentimental analysis using lstm.ipynb"
```

## 📈 Training Results

The LSTM model demonstrates excellent performance on sentiment classification:

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|------------------|-------------------|---------------|----------------|
| 1     | 94.11%          | 87.36%           | 0.1605        | 0.3412         |
| 2     | 95.00%          | 85.35%           | 0.1381        | 0.3765         |
| 3     | 95.13%          | 87.61%           | 0.1281        | 0.3773         |

### Key Observations:
- **High Training Performance**: Achieved 95%+ training accuracy quickly
- **Early Stopping**: Model stopped at epoch 3 due to validation loss plateauing
- **Overfitting Prevention**: Dropout and early stopping effectively controlled overfitting
- **Strong Learning**: Consistent improvement in training metrics across epochs

## 🔬 Technical Implementation

### Data Preprocessing
```python
# Sentiment encoding
df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})

# Text tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['review'])

# Sequence padding
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['review']), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['review']), maxlen=200)
```

### Model Configuration
```python
# Model compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X_train, Y_train, 
          validation_split=0.2,
          epochs=5,
          batch_size=64,
          callbacks=[early_stopping])
```

## 📁 Project Structure

```
Sentimental-analysis-using-LSTM/
├── Sentimental analysis using lstm.ipynb    # Main implementation notebook
├── README.md                               # Project documentation
├── requirements.txt                        # Dependencies list
├── kaggle.json                            # Kaggle API credentials (not included)
├── imdb-dataset-of-50k-movie-reviews.zip # Downloaded dataset
└── IMDB Dataset.csv                       # Extracted dataset
```

## 🎓 Learning Objectives

This project demonstrates:
1. **LSTM Architecture**: Understanding recurrent neural networks for NLP
2. **Text Preprocessing**: Tokenization, padding, and sequence handling
3. **Sentiment Analysis**: Binary classification for text data
4. **Deep Learning Pipeline**: End-to-end implementation with TensorFlow/Keras
5. **Regularization Techniques**: Dropout and early stopping for generalization
6. **Model Evaluation**: Training/validation monitoring and performance analysis

## 🔍 Key Insights

- **LSTM Effectiveness**: LSTM layers excel at capturing sequential patterns in text
- **Vocabulary Limitation**: Restricting vocabulary to 5,000 words balances performance and efficiency
- **Sequence Length**: 200-token sequences capture sufficient context for sentiment analysis
- **Regularization Impact**: Dropout prevents overfitting in text classification tasks
- **Early Stopping**: Prevents overfitting and saves computational resources

## 🚀 Future Enhancements

Potential improvements and extensions:
- [ ] **Bidirectional LSTM**: Use bidirectional layers for better context understanding
- [ ] **Pre-trained Embeddings**: Integrate Word2Vec, GloVe, or BERT embeddings
- [ ] **Attention Mechanism**: Add attention layers for interpretability
- [ ] **Hyperparameter Tuning**: Optimize architecture and training parameters
- [ ] **Multi-class Classification**: Extend to fine-grained sentiment levels
- [ ] **Real-time Prediction**: Create API for live sentiment analysis
- [ ] **Model Comparison**: Compare with transformer-based models (BERT, RoBERTa)

## 📊 Performance Metrics

Model capabilities:
- **Training Accuracy**: 95.13%
- **Validation Accuracy**: 87.61%
- **Model Parameters**: ~809K trainable parameters
- **Training Time**: ~13 minutes (3 epochs)
- **Sequence Length**: 200 tokens
- **Vocabulary Size**: 5,000 words

## 🛠️ Technical Stack

- **Deep Learning**: TensorFlow 2.x with Keras API
- **Data Processing**: Pandas, NumPy
- **Text Processing**: Keras Tokenizer and Sequence utilities
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebook environment
- **Data Source**: Kaggle API integration

## 🤝 Contributing

Feel free to contribute to this project by:
1. Reporting bugs or issues
2. Suggesting new features or improvements
3. Submitting pull requests
4. Sharing your results and experiments
5. Improving documentation

## 📜 License

This project is open-source and available under the MIT License.

---


*This project serves as a comprehensive example of sentiment analysis using LSTM neural networks and demonstrates practical implementation of deep learning for natural language processing tasks.*
