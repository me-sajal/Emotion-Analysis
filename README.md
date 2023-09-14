
# Emotion Dataset Analysis
The process for classifying text in the Emotion Dataset, which is based upon various algorithms and models of neural network, is described in this comprehensive documentation. The objective is to classify the emotions which have been reported in text comments and assess how different approaches are performing.

In Natural Language Processing, it is a key task to classify text according to the predefined categories or labels that are included in these documents. In this project, we're focusing on the classification of emotions in textual messages by traditional machine learning algorithms and neural network models.
### Count Plot of Emotion labels
![](https://github.com/me-sajal/Emotion-Analysis/blob/main/evaluation_outputs_images/count.png)

## Data Loading and Preprocessing
 Training, testing and validation data are presented in a dataset that is accompanied by comments matching the appropriate emotion labels. The preprocessing steps include:

* Reading and loading data in the files,
* Removing stopwords from text,
* Text Vectorization.


# Traditional Machine Learning Models
## Multinomial Naive Bayes
Multinomial Naive Bayes is one of the most popular supervised learning classifications that is used for the analysis of the categorical text data. Text data classification is gaining popularity because there is an enormous amount of information available in email, documents, websites, etc. that needs to be analyzed.

## Logistic Regression
Similar to the Naive Bayes model, Logistic Regression is trained and evaluated using TF-IDF features. Accuracy and classification metrics are computed. Logistic regression is used to predict the categorical dependent variable. It's used when the prediction is categorical, for example, yes or no, true or false, 0 or 1. For instance, insurance companies decide whether or not to approve a new policy based on a driver's history, credit history and other such factors.
![](https://miro.medium.com/v2/resize:fit:1400/1*dm6ZaX5fuSmuVvM4Ds-vcg.jpeg)

## Support Vector Machines (SVM)
An SVM model with a linear kernel is trained and evaluated on the TF-IDF transformed data. Support Vectors. Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier.
![](https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm5.png)

# Neural Network Model
## LSTM-Based Text Classification
An LSTM-based neural network is built for text classification. The text data is preprocessed using a TextVectorization layer, and the model architecture consists of an embedding layer, bidirectional LSTM layers, dense layers, and a softmax output layer
![](https://spotintelligence.com/2023/01/11/lstm-in-nlp-tasks/)

# Model Comparison and Evaluation
To compare the performance of the models, we evaluate them using various metrics:

### Multinomial Naive Bayes:
* Test Accuracy: 0.691

*  Confusion matrix
![](https://github.com/me-sajal/Emotion-Analysis/blob/main/evaluation_outputs_images/nb%20cm.png?raw=true)

### Logistic Regression:
* Test Accuracy: 0.871
* Confusion matrix
![](https://github.com/me-sajal/Emotion-Analysis/blob/main/evaluation_outputs_images/lr%20cm.png?raw=true)

### Support Vector Machines (SVM):
* Test Accuracy: 0.886
* Confusion matrix
![](https://github.com/me-sajal/Emotion-Analysis/blob/main/evaluation_outputs_images/svm%20cm.png?raw=true)

#### LSTM (Neural Network):
* Test Loss: 0.278

* Test Accuracy: 0.902

Training history plot showing train loss and validation loss  per epoch.
![](https://github.com/me-sajal/Emotion-Analysis/blob/main/evaluation_outputs_images/train_val_loss.png?raw=true)

Training history plot showing train accuracy and validation accuracy per epoch.
![](https://github.com/me-sajal/Emotion-Analysis/blob/main/evaluation_outputs_images/train_val_accuraacy.png?raw=true)

* Confusion Matrix
![](https://github.com/me-sajal/Emotion-Analysis/blob/main/evaluation_outputs_images/lstm%20cm.png?raw=true)


# Conclusion
The provided documentation is describing a detailed process of performing text classification on a dataset called the "Emotions Dataset." The goal of this text classification task is to classify text data into different emotion categories. The document outlines two different approaches for achieving this task: traditional machine learning algorithms and a neural network model.

1. **Traditional Machine Learning Algorithms**: This approach likely involves using classical machine learning techniques such as decision trees, random forests, support vector machines, or Naive Bayes to classify text data into emotional categories. The document explains the step-by-step process of implementing these algorithms, including data preprocessing, feature extraction, model training, and evaluation.

2. **Neural Network Model (LSTM-based)**: In contrast to traditional machine learning, this approach utilizes a neural network, specifically mentioning that it's based on Long Short-Term Memory (LSTM), a type of recurrent neural network. LSTMs are well-suited for sequence data, making them suitable for text classification tasks. The document goes on to detail the architecture of this neural network, including the number of layers, units, and other relevant parameters.

The documentation then discusses the comparative evaluation of these two approaches based on their performance. Specifically, it mentions that the LSTM-based neural network outperforms the traditional machine learning algorithms in terms of accuracy and overall emotion classification. This suggests that the neural network approach is more effective at correctly classifying text data into emotion categories.

The document promises to provide a comprehensive explanation of various aspects:

- **Code Explanation**: It likely includes a breakdown of the code used for both traditional machine learning algorithms and the neural network. This helps readers understand how the models were implemented.

- **Different Models Employed**: It explains the specific machine learning algorithms and the neural network architecture used in the classification task. This information is crucial for understanding the methods applied.

- **Performance Metrics**: The document should elaborate on the metrics used to evaluate the models. Common metrics for text classification include accuracy, precision, recall, F1-score, and possibly others. Understanding these metrics helps assess model performance objectively.

- **Comparative Analysis**: The comparative analysis is vital for understanding why one approach outperformed the other. It may delve into the strengths and weaknesses of each method, discussing scenarios where one approach might be more suitable than the other.

In summary, this documentation serves as a detailed guide for performing text classification on the Emotions Dataset, offering insights into the implementation of traditional machine learning algorithms and a neural network model. It also highlights the superior performance of the neural network and provides in-depth explanations, code, and performance metrics to support the findings. This documentation can be valuable for researchers or practitioners working on text classification tasks and seeking a comprehensive understanding of the process and results.

