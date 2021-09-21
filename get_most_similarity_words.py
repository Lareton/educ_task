# импортируем нужные библиотеки
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import gensim.downloader as api
import nltk
import string
import pymorphy2
import pandas as pd
import numpy as np
import json

# загрузим предобученную word2vec модель

word2vec = api.load("word2vec-ruscorpora-300")
word2vec.key_to_index = {k[:k.find('_')]: v for k, v in word2vec.key_to_index.items()}

# подключим библиотеки для лемматизации текста и удаления стоп-слов
morph = pymorphy2.MorphAnalyzer()
nltk.download('stopwords')
nltk.download('punkt')

stopWordsRu = set(stopwords.words("russian"))
punctuation = set(string.punctuation + "—" + "«" + "»" + "1234567890")


def validateWord(word: str) -> bool:
    """
    Функция для проверки, ялвяются ли все символы слова валидными.
    Принимает: str, какое-то слово;
    Возвращает: bool, True если слово валидное, иначе False.
    
    Пример использования:
        >> validateWord('?') # False
        >> validateWord('hi') # True
    """
    if word in stopWordsRu or word in punctuation:
        return False
    allSymValid = False

    for sym in word:
        if not sym in punctuation:
            allSymValid = True
            break

    return allSymValid


def prepareText(text: str) -> list:
    """
    Функция для предобработки текста перед подачей в w2v модель.
    Удаляет стоп-слова, лемматизирует все слова в тексте, а также разбивает его на список слов
    Принимает: str, текст;
    Возвращает: list, список лемматизированных валидных слов.
    
    Пример использования:
        >> prepareText("Придумай название для парнокопытных и пресмыкающихся, собранных вместе.")
        # ['придумать', 'название', 'парнокопытный', 'пресмыкаться', 'собранный', 'вместе']
    """
    text.replace("ё", "е")
    text.replace("\n", " ")

    out = []
    for word in nltk.word_tokenize(text):
        word = word.strip()
        if not validateWord(word): # если слово не валидное
            continue
        for p in punctuation: # удалим все знаки пунктуации
            word = word.replace(p, "")
        out.append(morph.parse(word.lower())[0].normal_form) # приведём слово к нормальной форме
    return out


def getMostSimilarityWords(text:str, theme:str, n:int=3) -> list:
    """
    Функция для вычисления самых близко-относящихся к заданной теме слов в тексте.
    Принимает:
        -text: str, текст, в котором мы ищем наиболее близкоотносящиеся к теме слова;
        -theme: str, тема, к которой мы ищем наиболее близкоотносящиеся слова;
        -n: int, количество интересующих нас наиболее бликоотносящихся слов.
    Возвращает:
        -mostSimilarityWords: list длинной в n с самыми близкоотносящимися к theme словами в строке text.
    Пример использования:
        >>
    """
    theme = prepareText(theme)[0]
    wordSimilarity = {}
    for word in prepareText(text):
        try:
            similarity = word2vec.similarity(word, theme)
        except: # если слова нет в словаре word2vec модели
            similarity = 0
        wordSimilarity[word] = similarity
        
    wordSimilaritySorted = dict(sorted(wordSimilarity.items(), key=lambda item: item[1], reverse=True))
    return list(wordSimilaritySorted.keys())[:min(len(wordSimilaritySorted), n)]