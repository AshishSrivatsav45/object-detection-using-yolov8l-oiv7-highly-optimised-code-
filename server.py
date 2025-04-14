from flask import Flask, request, jsonify, render_template


from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable requests from our local frontend

latest_text = "Waiting for data..."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update_text():
    global latest_text
    latest_text = request.json.get("text", "")
    return jsonify({"status": "success", "received": latest_text})

@app.route('/get', methods=['GET'])
def get_text():
    return jsonify({"text": latest_text})

if __name__ == '__main__':
    # Listen on all interfaces so you can access it from your phone
    app.run(host='0.0.0.0', debug=True)
