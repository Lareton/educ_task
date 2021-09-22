import telebot
from messages import ERROR_MESSAGE, START_MESSAGE, ABOUT_US_MESSAGE, INSTRUCTION_MESSAGE
from classify_theme import BertClassifier
from get_most_similarity_words import getMostSimilarityWords

TOKIN = "2002168472:AAGqtBG-7EqRzPB1pwBQrE79-c0svzcOf4c"
bot = telebot.TeleBot(TOKIN)

# обрабатываем команду /start
@bot.message_handler(commands=['start'])
def start_message(message):
    keyboard = telebot.types.ReplyKeyboardMarkup(True)
    keyboard.row('Инструкция', 'О нас')
    bot.send_message(message.chat.id, START_MESSAGE, reply_markup=keyboard)

# обрабатываем остальные команды и запрос на классификацию
@bot.message_handler(content_types=['text'])
def send_text(message):
    print(f"got message: {message}")

    # обрабатываем сообщение
    if message.text.lower() == 'инструкция':
        bot.send_message(message.chat.id, INSTRUCTION_MESSAGE, parse_mode="Markdown")
    elif message.text.lower() == 'о нас':
        bot.send_message(message.chat.id, ABOUT_US_MESSAGE, parse_mode="Markdown", disable_web_page_preview=True)
    else:
        if message.text[0] == "/":
            bot.send_message(message.chat.id, ERROR_MESSAGE,  parse_mode="Markdown")
        else:
            bot.send_message(message.chat.id, """Классификация идет... Это может занять некоторое время\n(около 5 секунд)""")

            # формируем предсказание
            task = message.text
            text_classifier = BertClassifier("theme_classifier_weights.pt")
            theme, accuracy = text_classifier.predict(task)
            similarity_words = getMostSimilarityWords(task, theme, max_count=7, max_difference=3.1)

            answer_predict = f"""*Категория данного текста*: {theme}\n*Уверенность*: {round(accuracy, 3)*100}%\n*Ключевые слова*: {", ".join(similarity_words)}"""
            bot.send_message(message.chat.id, answer_predict,  parse_mode="Markdown")

print("starting bot...")
bot.polling(none_stop=True, interval=0)
