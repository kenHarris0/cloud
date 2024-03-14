from flask import Flask, render_template, request, jsonify
import openai
import requests
from io import BytesIO
import pdfplumber
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

app = Flask(__name__)
# Including OpenAI API key directly
openai.api_key = ''

# Default PDF file
DEFAULT_PDF_FILE = "https://project-web-sjce.s3.amazonaws.com/RULES+word.pdf"

# Function to extract text from PDF


def extract_text_from_pdf(pdf_file):
    if pdf_file.startswith('http://') or pdf_file.startswith('https://'):
        response = requests.get(pdf_file)
        response.raise_for_status()  # Ensure the request was successful
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() if page.extract_text() else ''
    else:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() if page.extract_text() else ''
    return text

# Function to summarize text (optional)


def summarize_text(text, ratio=0.3):  # Adjust ratio for desired summary length
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=int(
        ratio * len(parser.document.sentences)))
    summary_text = " ".join([str(sentence) for sentence in summary])
    return summary_text

# Function to answer questions using OpenAI


def answer_question(question, context, max_tokens=150):  # Reduce answer length by default
    shortened_context = context[:4096]  # Limit context length to 4096 tokens
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"{shortened_context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].text.strip()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            question = request.form['question']

            # Extract text from the default PDF file
            pdf_text = extract_text_from_pdf(DEFAULT_PDF_FILE)

            # Optional: summarize the PDF text before asking questions
            # pdf_text = summarize_text(pdf_text)

            answer = answer_question(question, pdf_text)

            return jsonify({'question': question, 'answer': answer})
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
