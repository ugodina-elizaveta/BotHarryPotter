from functools import lru_cache


@lru_cache(maxsize=1000)
def get_response(model, input_text, sentences):
    """
    Получает ответ от модели на основе входного текста.
    :param model: Обученная модель RetrievalModel.
    :param input_text: Входной текст (строка).
    :param sentences: Список предложений (list или tuple).
    :return: Ответ (строка).
    """
    index = model.predict([input_text])[0]
    return sentences[index]