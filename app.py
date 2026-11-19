from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Завантажуємо реальну попередньо навчену ML-модель для аналізу емоцій (HuggingFace)
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# Словник для перекладу емоцій, підбору емодзі та порад українською
EMOTION_DATA = {
    "joy": {
        "label": "Радість",
        "emoji": "😄",
        "advice": "Чудовий настрій! Зафіксуй цей момент і поділися позитивом з друзями — сьогодні ідеальний день для нових звершень."
    },
    "sadness": {
        "label": "Сум",
        "emoji": "😢",
        "advice": "Виглядає, що ви сумуєте — ось цитата для натхнення: «Навіть найтемніша ніч закінчується світанком». Зроби паузу та випий теплого чаю."
    },
    "anger": {
        "label": "Гнів / Роздратування",
        "emoji": "😠",
        "advice": "Зроби глибокий вдих і видих. Спробуй перемкнутися на коротку прогулянку або фізичну активність, щоб скинути напругу."
    },
    "fear": {
        "label": "Тривога / Страх",
        "emoji": "😨",
        "advice": "Не хвилюйся завчасно. Розбий складне завдання на маленькі кроки — контроль над деталями швидко поверне спокій."
    },
    "surprise": {
        "label": "Здивування",
        "emoji": "😲",
        "advice": "Несподіванки роблять життя цікавішим! Використай цю нову інформацію на свою користь."
    },
    "disgust": {
        "label": "Відраза / Неприйняття",
        "emoji": "🤢",
        "advice": "Твоя інтуїція чітко підказує, що тобі не підходить. Просто дистанціюйся від неприємної ситуації."
    },
    "neutral": {
        "label": "Спокій / Нейтральний стан",
        "emoji": "😐",
        "advice": "Рівний та врівноважений стан — найкращий фундамент для продуктивної роботи та навчання."
    }
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("user_text", "").strip()
        if user_text:
            # Проганяємо текст через ML-модель
            prediction = emotion_classifier(user_text)[0][0]
            raw_emotion = prediction["label"]
            confidence = round(prediction["score"] * 100, 1)

            info = EMOTION_DATA.get(raw_emotion, EMOTION_DATA["neutral"])
            result = {
                "emotion": info["label"],
                "emoji": info["emoji"],
                "advice": info["advice"],
                "confidence": confidence
            }

    return render_template("index.html", result=result, user_text=user_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
