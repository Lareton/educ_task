
import gensim.downloader as api
word2vec_model300 = api.load('word2vec-ruscorpora-300')


words = list(word2vec_model300.index_to_key)
lang_dct = {}
full_lang_dct = {}
words_dop = [i[:i.find('_')] for i in words]

for i in range(len(words)):
    lang_dct[words_dop[i]] = words[i]

for i in lang_dct.keys():
    full_lang_dct[lang_dct[i]] = i



def simular(word, count):
    tmp = lang_dct[word]
    return [full_lang_dct[i[0]] for i in word2vec_model300.most_similar(tmp, topn=count)]


def get_all_key_words(theme):
    res = []
    for i in simular(theme, 3):
        for j in simular(i, 2):
            res.append(j)
        res.append(i)
    return res

print(get_all_key_words("спорт"))
