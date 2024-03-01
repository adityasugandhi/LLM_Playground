from flask import Flask, jsonify, request
from Inferencer import Inferncer
from dataloader import DataLoader

app = Flask(__name__)

UPLOAD_FOLDER = './data/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

inferencer = Inferncer()
data_loader = DataLoader()

# Initialize chroma_store as a global variable
# chroma_store = data_loader.dataloader()
# in_memory_store = data_loader.InMemory_dataloader()
chroma_store = None
in_memory_store = None
@app.route("/")
def home():
    return "Welcome to the Flask app!"

@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return jsonify({"message": "File uploaded successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/sync", methods=["POST"])
def sync_and_run_dataloader():
    global chroma_store 
    global in_memory_store# Access the global chroma_store variable
    try:
        # Optionally, you can add authentication or other checks here

        # Call the dataloader function
        chroma_store = data_loader.dataloader()
        in_memory_store = data_loader.InMemory_dataloader()

        return jsonify({"message": "DataLoader executed successfully", "result": "success"})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        query = data.get("question", "")
        model = data.get("model", "")

        if chroma_store is None:
            return jsonify({"error": "Chroma store not initialized. Run sync_and_run_dataloader first."})

        if model == "OpenAI":
            results = inferencer.OpenAI(query=query)
            return jsonify({"results": results})
        elif model == "LlamaCpp":
            results = inferencer.LlamaCpp(query=query)
            return jsonify({"results": results})
        else:
            return jsonify({"error": f"Invalid model specified: {model}"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)