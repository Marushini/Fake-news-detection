Fake News Detection Using Python
1. Introduction
In the digital age, the rapid spread of misinformation has posed significant challenges. This project explores how machine learning can be applied to detect fake news. Using Logistic Regression and TF-IDF vectorization, the model analyzes the text content of news articles and classifies them as either 'Real' or 'Fake'.

2. Prerequisites
Before running the code, ensure the following libraries are installed:
 pandas
 scikit-learn
 nltk
 seaborn
 matplotlib
 
3.Dataset
The dataset used in this project is a CSV file (`News.csv`) with two key columns:
  text: The news article's content.
  class: The label indicating whether the article is 'Real' (1) or 'Fake' (0).

4. Methodology
4.1 Data Preprocessing
To enhance accuracy, stopwords (common words such as 'the', 'is') are removed from the text. NLTK's stopwords list is used for this purpose. Each article's text is cleaned to ensure only relevant words remain.
4.2 Train-Test Split
The dataset is split into training and testing sets. 80% of the data is used for training the model, while the remaining 20% is reserved for evaluation.
4.3 Text Vectorization
The cleaned text is transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency). This technique assigns higher weights to words that are unique to individual articles, while downweighting common words.
4.4 Model Training
A Logistic Regression model is employed to classify the articles. The model is trained on the transformed TF-IDF data to learn the patterns that distinguish real news from fake news.
4.5 Model Evaluation
After training, the model's performance is evaluated using the testing set. Key metrics such as accuracy, precision, recall, and F1-score are computed. A confusion matrix is also generated to visualize the classification results.
4.6 Visualization
The confusion matrix is presented as a heatmap to provide a clear depiction of the model's predictions. The matrix shows the number of correctly and incorrectly classified articles in both 'Real' and 'Fake' categories.

5. Results
The model achieves a satisfactory level of accuracy, demonstrating its ability to effectively distinguish between real and fake news. The detailed classification report and confusion matrix offer insights into the model's strengths and areas for improvement.
6. Conclusion
This project illustrates the potential of machine learning in combating the spread of misinformation. While the current model performs well, further enhancements such as incorporating advanced algorithms or larger datasets could improve its accuracy and robustness.
7. References
1. Pandas Documentation: https://pandas.pydata.org/docs/
2. Scikit-learn Documentation: https://scikit-learn.org/stable/
3. NLTK Documentation: https://www.nltk.org/
4. Seaborn Documentation: https://seaborn.pydata.org/
5. Matplotlib Documentation: https://matplotlib.org/
