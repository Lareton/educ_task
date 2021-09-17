from key_word_function import get_all_key_words
import nltk
import gensim.downloader as api
import pymorphy2
import pandas as pd
from nltk.corpus import stopwords
import string
import telebot
import logging
import csv
root = logging.getLogger()
root.setLevel(logging.DEBUG)

morph = pymorphy2.MorphAnalyzer()
sample = pd.read_csv('test.csv')
word2vec_model300 = api.load('word2vec-ruscorpora-300')
# nltk.download('stopwords')
# nltk.download('punkt')

good_words = []         # слова которые помогают определить категорию
best_words = []         # слова которые совпадают с конечной категорией
PREDICT = False         # нужно ли делать предикт на всех тестовых данных
COUNT_SMALL_TEST = 53   # кол-во в ручную размеченных данных
COUNT_BIG_TEST = 514   # кол-во в ручную размеченных данных


words = list(word2vec_model300.index_to_key)
words_dop = [i[:i.find('_')] for i in words]

print(words[:100])
print(words_dop[:100])
print(len(words))

lang_dct = {}

for i in range(len(words)):
    lang_dct[words_dop[i]] = words[i]


for i in lang_dct:
    print(i, lang_dct[i])
    if i == "самый":
        break

themes_raw = ["спорт", "музыка", "литература", "животные"]
themes = [lang_dct[i] for i in ["спорт", "музыка", "литература", "животное"]]

stopWordsRu = set(stopwords.words("russian"))
punctuation = set(string.punctuation + "—")


def validateWord(word):
    if word in stopWordsRu or word in punctuation:
        return False
    allSymValid = False

    for sym in word:
        if not sym in punctuation:
            allSymValid = True
            break

    return allSymValid


def prepareText(text: str) -> list:
    text.replace("ё", "е")
    text.replace("\n", " ")

    out = []
    for word in nltk.word_tokenize(text):
        word = word.strip()
        if not validateWord(word):
            continue
        out.append(morph.parse(word.lower())[0].normal_form)
    return out


def mean_distances(word, words_thems):
    a = word2vec_model300.distances(lang_dct[word], [lang_dct[i] for i in words_thems])
    #print(f"check {word} to {words_thems}        get {a}")
    a.sort()
    a = a[:min(3, len(a))]
    return sum(a) / len(a)



words_thems = dict()
words_thems["спорт"] = ["спорт"] * 20 + ["футбол", "мяч", "велосипед"] + ["волейбол"] + ["бежать"] + ["плыть"] + ["грести"] + ["прыжок", "сдавать", "ролик"]
words_thems["музыка"] = ["музыка"] * 20 + ["скрипка", "флейта", "гитара", "пианино"]
words_thems["литература"] = ["литература"] * 20 +  ["писатель", "написать", "сочинение", "герой", "произведение", "книга", "бумага", "газета"]
words_thems["животные"] = ["животное"] * 20 + ['собака', 'тигр', 'лев', 'жираф', 'дельфин', 'акула', 'кошка', 'бабочка', 'насекомое', 'млекопитающее', 'змея', "кролик", "удочка", "пчела"]

# for i in  themes_raw[:-1]:
#     words_thems[i] *= 2
#     words_thems[i] += get_all_key_words(i)
#     print(words_thems[i])
# words_thems["животные"] = get_all_key_words("животное")


for i in words_thems.keys():
    tmp = []
    for j in words_thems[i]:
        for k in get_all_key_words(j):
            tmp.append(k)

    for j in tmp:
        words_thems[i].append(j)


print()
print(words_thems)
print()

def mean_distances_to_themes(word):
    res = []
    for theme in ["спорт", "музыка", "литература", "животные"]:
        tmp = mean_distances(word, words_thems[theme])
        if tmp != 2:
            res.append(tmp)
    return res


def main_answer(text):
    text = prepareText(text)
    sames = []
    for word in text:
        key = []
        key.append(word)
        try:
            key += mean_distances_to_themes(word)
            sames.append(key)
        except Exception as e:
                # print(f"EROROOROORO {word} -----       {e}")
                pass
    return sames


def main_function(text):
    global good_words, best_words

    text = text.replace("фискультуре", "физкультура")
    if "физкультура" in text:
        print("a" * 100)
    k = main_answer(text)
    text = text.split()
    print("text", text)
    important_cases = []

    for i in k:
        # print(i, " "*10, max(i[1:]) - min(i[1:]))
        important_cases.append((i, max(i[1:]) - min(i[1:])))

    main_res = [0, 0, 0, 0]

    important_cases.sort(key=lambda x: x[1], reverse=True)
    # print(important_cases)

    important_cases = important_cases[:30]

    good_words = [i[0] for i in important_cases]

    important_cases = [i[0] for i in important_cases]

    print("Помогли слова ", [i[0] for i in important_cases])

    for i in important_cases:
        for j in range(4):
            main_res[j] += i[j + 1]

    if  len(important_cases) == 0:
        return -1

    for i in range(4):
        main_res[i] /= len(important_cases)
        if len(important_cases) == 0:
            print("AAAAAAA" * 25)

    print(main_res)
    print(main_res.index(min(main_res)))
    print(themes_raw[main_res.index(min(main_res))])
    best_words = []

    for i in good_words:
        tmp = i[1:]
        if tmp.index(min(tmp)) == main_res.index(min(main_res)):
            best_words.append(i[0])

    return main_res.index(min(main_res))


my_answers = []
results_answers = [["id", "category", "keywords"]]
if PREDICT:
    count_tests = COUNT_BIG_TEST
else:
    count_tests = COUNT_SMALL_TEST

for ind, i in enumerate(sample["task"][:count_tests]):
    my_answers.append(main_function(i))
    print("answer: ", themes_raw[my_answers[-1]], list(set(best_words[:3])))
    results_answers.append([ind, themes_raw[my_answers[-1]], ";".join(list(set(best_words[:3])))])

    for i in range(5):
        print()

print(results_answers)

if PREDICT:
    data = results_answers

    with open('sub1.csv', 'w', encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        for row in data:

            writer.writerow(row)

    print("ALL WRITED")


answers = [0, 3, 0, 3, 0, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 2, 0, 2, 3, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 2, 3, 0, 1, 2, 0, 0, 0, 0, 2, 3, 3, 2]

print(answers)
print(my_answers)


wrong_answers = 0
for i in range(COUNT_SMALL_TEST):
    print(i)
    if answers[i] == my_answers[i]:
        print("ok", i, sample["task"][i][:100].replace("\n", " "), answers[i])
    else:
        wrong_answers += 1
        print("noooo", i)
        print(sample["task"][i].replace("\n", ""), "----"*5, my_answers[i], answers[i])
print(wrong_answers)


bot = telebot.TeleBot('1454441673:AAE4jugofHCscmt5-ufg_whJ4dd471OJodM')


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    print(message.text)
    text = message.text
    text.lower()
    answer = main_function(text)
    if answer == -1:
        bot.send_message(message.from_user.id, "Слишком короткое предложение невозможно определить смысл")
    else:
        bot.send_message(message.from_user.id, themes_raw[answer])
        bot.send_message(message.from_user.id, f"мне помогли слова {best_words}")
logging.info(" start bot....")
bot.polling(none_stop=True, interval=0)
