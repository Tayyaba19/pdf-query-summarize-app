## Overview

This project is a PDF querying and summarization tool that allows users to upload a PDF file and either ask questions or generate a summary from the document's contents. The app uses **Chainlit** for the user interface and integrates a language model (Mistral for chating and Flant5 for summery) for natural language processing.

## Features

### 1. PDF Querying
- Users can upload a PDF file.
- The system extracts the text and allows users to ask queries about the document.
- An open-source language model is used to process the queries and provide accurate answers.

### 2. PDF Summarization
- Users can summarize the entire document after uploading.
- The summary is generated using the same language model that processes the queries.

### User Interface
- Built using **Chainlit**, providing a simple and user-friendly interface:
  - A file upload button for the PDF.
  - A text box for user queries.
  - A button to generate the summary.
  - Results are displayed clearly in the chat interface.

## Setup Instructions

### 1. Clone the Repository
git clone https://github.com/your-repo-url/pdf-query-summarize-app.git
cd pdf-query-summarize-app

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Setup Environment Variables
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACEHUB_API_TOKEN=your_huggingface-api-token

### 4. Running the Application
chainlit run app.py for openai model
chainlit run os_models.py for opensource models


This will launch a local server where you can upload PDF files, ask queries, and generate summaries.

### 5. PDF Upload & Usage
Once the app is running, upload a PDF by clicking the "Upload" button.
Enter queries in the provided text box to get specific answers from the PDF.
Click "Summarize" to get a concise summary of the document.
Technical Considerations
Text Extraction: The PDF text is extracted using the all-MiniLM-L6-v2 model from hugginhface.
Model Integration: The app integrates Mistral model for answering queries
Performance: The app ensures fast and accurate query responses by caching the document text and reusing the same model instance.
Additional Features
The application is modular and easy to extend for more advanced features.
Summarization and querying are processed separately to enhance flexibility.
Requirements

An API key from OpenAI or any supported model provider.
HUGGINGFACEHUB_API_TOKEN from huggingface by logging in to your account.
