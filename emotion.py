import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords
import string
from sklearn.svm import SVC

class EmotionAnalyzer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
        self.model = MultinomialNB()
        self.classifier = SVC(kernel='linear', random_state=10)

    def preprocess_text(self, text):
        return text.lower()
    
    def clean_twitter_text(self,text):
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'RT[\s]+', '', text)
        text = re.sub(r'https?://\S+', '', text)

        text = re.sub(r'[^A-Za-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def load_and_prepare_data(self, filepath):
        self.data = pd.read_csv(filepath)
        self.data = self.data[["label","tweet"]]
        self.data = self.data.drop_duplicates(subset=["tweet"])
        self.data = self.data.dropna()
        self.data['tweet'] = self.data['tweet'].apply(self.clean_twitter_text)
        self.data['tweet'] = self.data['tweet'].str.lower()
        self.data['review_len'] = self.data['tweet'].apply(lambda x: len(str(x)) - str(x).count(" "))

        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        all_stopwords = stopwords.words('indonesian')

        lemmatizer = nltk.stem.WordNetLemmatizer()
        self.data['tokens'] = self.data['tweet'].apply(lambda x: self.tokenize_text(x))
        self.data['lemmatized_review'] = self.data['tokens'].apply(
            lambda x: self.lemmatize_text(x, lemmatizer, all_stopwords)
        )


    @staticmethod
    def lemmatize_text(token_list, lemmatizer, stopwords):
        return " ".join([lemmatizer.lemmatize(token) for token in token_list if token not in set(stopwords)])
    @staticmethod
    def tokenize_text(text):
        return text.split()
    
    @staticmethod
    def count_punct(review):
        count = sum([1 for char in review if char in string.punctuation])
        return round(count / (len(review) - review.count(" ")), 3) * 100
    
    def train(self, test_size=0.7, random_state=800):
        X = self.data[['lemmatized_review', 'review_len']]
        y = self.data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        tfidf_train = self.tfidf.fit_transform(X_train['lemmatized_review'])
        tfidf_test = self.tfidf.transform(X_test['lemmatized_review'])

        self.classifier.fit(tfidf_train, y_train)
        score = self.classifier.score(tfidf_test, y_test)
        print(f"Model accuracy: {score:.2f}")

    def predict(self, comment):
        data = [comment]
        vect = self.tfidf.transform(data).toarray()
        prediction = self.classifier.predict(vect)
        return prediction[0]
