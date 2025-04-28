import os
import re
import json
import cohere
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

app = Flask(__name__)
CORS(app)

# Load your Q&A training data
with open('trained-data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json['message'].lower()

    # Emergency fund calculation
    expense_match = re.search(r'₹?(\d{4,6})', user_msg)
    months_match = re.search(r'(\d{1,2})\s*(months|month)', user_msg)

    if expense_match and months_match:
        expense = int(expense_match.group(1))
        months = int(months_match.group(1))
        total = expense * months
        return jsonify({"reply": f"Based on your input, your emergency fund should be ₹{total:,} (₹{expense:,} x {months} months)."})

    # Q&A matching
    user_vec = vectorizer.transform([user_msg])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    max_score = similarity[0][idx]

    if max_score > 0.4:
        return jsonify({'reply': answers[idx]})

    # Fallback: Use Cohere if no match
    try:
        response = co.generate(
            model='command-light',
            prompt=f"Answer this like a helpful financial assistant who only talks about emergency funds:\n\n{user_msg}",
            max_tokens=100,
            temperature=0.7
        )
        reply = response.generations[0].text.strip()
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'reply': f"Cohere error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
