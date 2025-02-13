from fastapi import FastAPI
from src.model import RetrievalModel
from src.inference import get_response
from src.preprocessing import preprocess_data, load_data
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class UserInput(BaseModel):
    user_input: str

# Подготавливаем данные для обучения модели
model = RetrievalModel()
train_df = preprocess_data(load_data('data/Friends.csv'))
model.fit(train_df['Text'])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Joey Bot API! Send a POST request to '/chat' with user_input JSON field for a response."}

@app.post("/chat")
def chat(user_input: UserInput):
    try:
        response = get_response(model, user_input.user_input, train_df)
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск приложения
# Используйте следующую команду для запуска сервиса:
# uvicorn main:app --reload
uvicorn.run(app, host="0.0.0.0", port=5000)