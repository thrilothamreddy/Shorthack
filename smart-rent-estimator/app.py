from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI  # ✅ For latest openai>=1.0.0

# ✅ Load environment variables and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load ML model and encoders
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
locality_encoder = joblib.load("locality_encoder.pkl")

@app.route('/')
def home():
    locality_list = list(locality_encoder.classes_)
    city_list = ['Kolkata', 'Hyderabad', 'Mumbai', 'Chennai', 'Delhi', 'Bangalore']
    return render_template('index.html', localities=locality_list, cities=city_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Mappings for categorical fields
        furnishing_map = {'Furnished': 2, 'Semi-Furnished': 1, 'Unfurnished': 0}
        tenant_map = {'Bachelors': 0, 'Family': 1, 'Bachelors/Family': 2}
        area_type_map = {'Super Area': 0, 'Carpet Area': 1}
        city_map = {'Kolkata': 0, 'Hyderabad': 1, 'Mumbai': 2, 'Chennai': 3, 'Delhi': 4, 'Bangalore': 5}

        # Prepare input for model
        input_data = [
            float(data['bhk']),
            float(data['size']),
            float(data['floor']),
            float(data['bathroom']),
            furnishing_map[data['furnishing']],
            tenant_map[data['tenant']],
            area_type_map[data['area_type']],
            city_map[data['city']],
            locality_encoder.transform([data['locality']])[0]
        ]

        prediction = model.predict([input_data])[0]

        # Reload UI with prediction
        locality_list = list(locality_encoder.classes_)
        city_list = list(city_map.keys())

        return render_template('index.html',
                               prediction=int(prediction),
                               localities=locality_list,
                               cities=city_list)

    except Exception as e:
        return f"❌ Error: {e}"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("question", "")

        if not user_input:
            return jsonify({"answer": "❗ Please enter a question."})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that supports rent estimation."},
                {"role": "user", "content": user_input}
            ]
        )

        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"})

# ✅ Entry Point
if __name__ == '__main__':
    app.run(debug=True)