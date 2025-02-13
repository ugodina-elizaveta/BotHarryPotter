cache = {}

def get_response(model, input_text, train_df):
    if input_text in cache:
        return cache[input_text]
    index = model.predict([input_text])[0]
    response = train_df.iloc[index]['Text']
    cache[input_text] = response

    return response