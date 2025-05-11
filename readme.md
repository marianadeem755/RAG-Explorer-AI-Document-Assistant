## 📄 RAG Document Assistant

The RAG Document Assistant is an AI powered application that allows users to upload documents and ask questions to receive intelligent answers. It utilizes Retrieval Augmented Generation (RAG) techniques to provide contextually relevant responses and supports translation capabilities for diverse languages.

## Features

- **Document Upload**: Upload PDF or TXT documents for analysis.
- **AI-Powered Q&A**: Ask questions based on the content of the uploaded documents.
- **Translation Support**: Translate answers into multiple languages including Urdu, Hindi, Spanish, and French.
- **Text-to-Speech**: Generate audio responses for the translated answers.
- **User -Friendly Interface**: Built with Streamlit for an interactive user experience.

## Installation
1. **Install the required packages**:
  ```
   pip install -r requirements.txt
  ```
2. **Set up your API key**:
   - Obtain your API key from Groq and set it as an environment variable
3. **Run the app locally**:
```
   streamlit run app.py
```
## Usage

1. **Upload a Document**: Use the file uploader to upload a PDF or TXT document.
2. **Ask Questions**: Enter your questions in the provided input field.
3. **Select Translation Options**: Choose a language for translation if needed.
4. **Receive Answers**: View the AI-generated answers and listen to audio playback if enabled.

## Acknowledgments

- **Streamlit**:Built with [Streamlit](https://streamlit.io/).
- **Sentence Transformers**: For embedding text and generating responses.
- **Deep Translator**: For translation capabilities.
- **gTTS**: For text-to-speech functionality.
