from classify_theme import BertClassifier
from get_most_similarity_words import getMostSimilarityWords

task = """Реши задачу. Сколько коробок 
          корма для собак поместится в ящике, 
          если все стороны ящика соответственно в 12 раз больше сторон коробки корма?"""

text_classifier = BertClassifier("theme_classifier_weights.pt")
theme = text_classifier.predict(task)
similarity_words = getMostSimilarityWords(task, theme)

print("Текст:", task)
print("Тема:", theme)
print("Помогли слова:", *similarity_words)