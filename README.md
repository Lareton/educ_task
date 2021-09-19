# educ_task
Education task for ML sber competition

ссылка на веса:  https://drive.google.com/file/d/1tXnLYCxs8Lgpg4YmlG8NsWNw1X-YB7BX/view?usp=sharing  

пример использования:  

```python
from classify_theme import BertClassifier  
from get_most_similarity_words import getMostSimilarityWords  

task = """Реши задачу. Сколько коробок   
          корма для собак поместится в ящике,  
          если все стороны ящика соответственно в 12 раз больше сторон коробки корма?"""  
 
text_classifier = BertClassifier("theme_classifier_weights.pt")  
theme = text_classifier.predict(task)  
similarity_words = getMostSimilarityWords(task, theme)

print("Тема:", theme) # животные
print("Помогли слова:", *similarity_words) # собака корм соответственно
