import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from emotion import EmotionAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


class SentimentAnalysis:
    def __init__(self, data_file, ignore_words_file, emotion_file):
        self.data_file = data_file
        self.ignore_words_file = ignore_words_file
        self.emotion_file = emotion_file
        self.ignore_words = set()
        self.data = None
        self.analyzer = EmotionAnalyzer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.analizing_method='stop_words'
        nltk.download('punkt')
        nltk.download('stopwords')
    
    def set_analize_method(self,method):
        self.analizing_method=method

    def load_ignore_words(self):
        try:
            with open(self.ignore_words_file, 'r') as file:
                self.ignore_words = set(word.strip().lower() for word in file.readlines())
        except FileNotFoundError:
            print(f"File {self.ignore_words_file} tidak ditemukan. Melanjutkan tanpa daftar ignore words.")

    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        if 'Comment' not in self.data.columns:
            raise ValueError("File CSV harus memiliki kolom bernama 'Comment'.")

    def train_emotion_analyzer(self):
        self.analyzer.load_and_prepare_data(self.emotion_file)
        self.analyzer.train()

    @staticmethod
    def clean_text(text, ignore_words):
        words = text.split()
        return ' '.join(word for word in words if word.lower() not in ignore_words)

    @staticmethod
    def load_lexicon(file_path):
        try:
            with open(file_path, 'r') as file:
                lexicon = set(word.strip().lower() for word in file.readlines())
            return lexicon
        except FileNotFoundError:
            print(f"File {file_path} tidak ditemukan. Pastikan file lexicon tersedia.")
            return set() 
    
    def analyze_sentiment_with_lexicon(self,text, lexicon):
        """
        Membersihkan teks dengan menghapus kata-kata yang tidak relevan
        berdasarkan lexicon yang diberikan.
        """
        words = word_tokenize(text.lower())  # Tokenisasi dan ubah teks menjadi huruf kecil
        cleaned_text = ' '.join(word for word in words if word in lexicon)
        print(cleaned_text)
        self.data.loc[self.data['Comment'] == text, 'clean_Comment'] = cleaned_text

        scores = self.sentiment_analyzer.polarity_scores(cleaned_text)
        polarity = scores['compound']
        subjectivity = scores['pos'] + scores['neg'] + scores['neu']
        emotion = self.analyzer.predict(cleaned_text)
        return polarity, subjectivity, emotion

    @staticmethod
    def sentimen_using_stop_words(text):
        stopwords_indonesia = set(stopwords.words('indonesian'))
        words = word_tokenize(text.lower())
        return ' '.join(word for word in words if word not in stopwords_indonesia)

    def analyze_sentiment_with_stop_words(self, text):
        cleaned_text = self.sentimen_using_stop_words(text)
        self.data.loc[self.data['Comment'] == text, 'clean_Comment'] = cleaned_text

        scores = self.sentiment_analyzer.polarity_scores(cleaned_text)
        polarity = scores['compound']
        subjectivity = scores['pos'] + scores['neg'] + scores['neu']
        emotion = self.analyzer.predict(cleaned_text)
        return polarity, subjectivity, emotion

    @staticmethod
    def categorize_sentiment(polarity):
        if polarity < 0:
            return 'Negatif'
        elif polarity == 0:
            return 'Netral'
        else:
            return 'Positif'

    def process_data(self):
        self.data['clean_Comment'] = self.data['Comment'].apply(
            lambda x: self.clean_text(str(x), self.ignore_words)
        )
        if self.analizing_method == 'stop_words':
            self.data[['Polarity', 'Subjectivity', 'Emotion']] = self.data['clean_Comment'].apply(
                lambda x: pd.Series(self.analyze_sentiment_with_stop_words(x))
            )
        else:
            lexicon = sa.load_lexicon('airin-ade.txt')
            self.data[['Polarity', 'Subjectivity', 'Emotion']] = self.data['clean_Comment'].apply(
                lambda x: pd.Series(self.analyze_sentiment_with_lexicon(x,lexicon))
            )
        self.data['Sentiment'] = self.data['Polarity'].apply(self.categorize_sentiment)

    def save_results(self, output_file):
        self.data.to_csv(output_file, index=False)
        print(f"Hasil analisis sentimen telah disimpan ke {output_file}")

    def visualize_wordcloud(self):
        all_words = ' '.join(self.data['clean_Comment']).split()
        word_counts = Counter(all_words)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud (Frekuensi Kemunculan Kata)')
        plt.show()

    def visualize_target_words(self, target_words):
        all_words = ' '.join(self.data['clean_Comment']).split()
        word_counts = Counter(all_words)
        target_word_counts = {word: word_counts.get(word, 0) for word in target_words}
        print(f"target_word_counts : {target_word_counts}")
        target_word_freq_df = pd.DataFrame(list(target_word_counts.items()), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
        sns.barplot(x='Frequency', y='Word', data=target_word_freq_df, palette='coolwarm', hue='Word', dodge=False, legend=False)
        plt.title('Frekuensi Kata Spesifik')
        plt.xlabel('Frekuensi')
        plt.ylabel('Kata')
        plt.show()

    def visualize_sentiments(self):
        sentiment_counts = self.data['Sentiment'].value_counts()
        categories = ['Negatif','Netral', 'Positif']
        for category in categories:
            if category not in sentiment_counts:
                sentiment_counts[category] = 0
        sentiment_counts = sentiment_counts[categories]
        print(f"sentiment_counts : {sentiment_counts}")
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm', hue=sentiment_counts.values, dodge=False, legend=False)
        plt.title('Distribusi Sentimen Berdasarkan Polarity')
        plt.xlabel('Kategori Sentimen')
        plt.ylabel('Jumlah')
        plt.show()

    def visualize_emotions(self):
        emotion_counts = self.data['Emotion'].value_counts()
        print(f"emotion_counts : {emotion_counts}")
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette='viridis', hue=emotion_counts.values, dodge=False, legend=False)
        plt.title('Distribusi Sentimen Berdasarkan Emosi')
        plt.xlabel('Kategori Emosi')
        plt.ylabel('Jumlah')
        plt.show()


if __name__ == "__main__":
    sa = SentimentAnalysis('data.csv', 'ignore-kata.txt', 'emotion.csv')
    sa.set_analize_method('stop_words')
    sa.load_ignore_words()
    sa.load_data()
    sa.train_emotion_analyzer()
    sa.process_data()
    sa.save_results('sentiment_analysis_results_3.csv')
    sa.visualize_wordcloud()
    sa.visualize_target_words(['airin', 'ade', 'rachmi', 'Diany', '01', '1', 'sumardi','02','2','andra','soni','dimyati','natakusuma'])
    sa.visualize_sentiments()
    sa.visualize_emotions()
