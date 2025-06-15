from flask import Flask, render_template, request, redirect, url_for# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

# Initialize the Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('home.html')

# Define the upload route where users can enter text or upload a file
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

# Define the summarize route where the text is summarized
@app.route('/summarize', methods=['POST'])
def summarize():
    # Initialize an empty string for the text input
    input_text = ""

    # Check if text was entered in the textarea
    if 'text' in request.form and request.form['text']:
        input_text = request.form['text']
    # Check if a file was uploaded
    elif 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        input_text = file.read().decode('utf-8')

    if not input_text:
        return redirect(url_for('upload'))

    # Tokenize and encode the input text
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary using the model
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return render_template('result.html', summary=summary)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)