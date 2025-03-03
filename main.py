from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from src.model import RetrievalModel
import logging
from src.inference import get_response
from src.preprocessing import preprocess_data, load_data

app = FastAPI()

# Загрузка и предобработка данных
df = preprocess_data(load_data('data/Harry_Potter_1.csv'))

# Обучение модели
model = RetrievalModel()
model.fit(df['Sentence'])


class ChatRequest(BaseModel):
    text: str
    character: str = "Harry"  # По умолчанию выбран Harry


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received request: {request}")
        # Предобработка данных для выбранного персонажа
        df_filtered = preprocess_data(load_data('data/Harry_Potter_1.csv'), speaker=request.character)
        # Преобразуем столбец 'Sentence' в список
        sentences = df_filtered['Sentence'].tolist()
        # Получаем ответ
        response = get_response(model, request.text, tuple(sentences))  # Используем tuple для хэширования
        logger.info(f"Response: {response}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}


# Запуск приложения
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
