import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import speech_recognition as sr
import gradio as gr

# Ensure necessary NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

# Load CSV dataset safely
def load_dataset(file_path="cpcsv.csv"):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python', encoding='utf-8')
        if 'query' not in df.columns or 'answer' not in df.columns:
            raise ValueError("CSV must contain 'query' and 'answer' columns.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame(columns=['query', 'answer'])

# Load the data
data = load_dataset()

# Intent detection keywords
GREETINGS = ["hello", "hi", "hey", "greetings"]
GOODBYES = ["bye", "goodbye", "see you", "farewell"]

# Convert audio to text
def convert_speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Speech service is unavailable. Please try again later."

# Get chatbot response
def get_response(user_input):
    if data.empty:
        return "Dataset is not loaded. Please check the CSV file."

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(user_input.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    if any(word in tokens for word in GREETINGS):
        return "Hello! How can I help you with child psychology today?"
    if any(word in tokens for word in GOODBYES):
        return "Goodbye! Take care and feel free to come back anytime."

    # Match based on content words
    for _, row in data.iterrows():
        question_tokens = word_tokenize(row['query'].lower())
        question_filtered = [word for word in question_tokens if word.isalpha() and word not in stop_words]
        if set(filtered_tokens) & set(question_filtered):
            return row['answer']

    return "I'm not sure how to answer that. You might want to consult a psychologist."

# Interface function
def chatbot_interface(text_input, audio_input):
    if audio_input is not None:
        user_query = convert_speech_to_text(audio_input)
    else:
        user_query = text_input

    if not user_query.strip():
        return "Please enter or say something."

    return get_response(user_query)

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("## 👶 Child Psychology Chatbot")

    with gr.Row():
        text_box = gr.Textbox(label="Type your question here")
        audio_upload = gr.Audio(type="filepath", label="Or upload an audio file")

    output_box = gr.Textbox(label="Chatbot's Response")

    text_box.submit(chatbot_interface, inputs=[text_box, audio_upload], outputs=output_box)
    audio_upload.change(chatbot_interface, inputs=[text_box, audio_upload], outputs=output_box)

    gr.Markdown("Upload your query as audio or type it above to receive a response.")

app.launch(share=True)
