import os
import chainlit as cl
from dotenv import load_dotenv
import logging
import warnings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.runnable import RunnablePassthrough

from langchain.chains import LLMChain, RetrievalQA
from transformers import pipeline
from documentLoader.file import DocumentLoader, DocumentLoaderException, configure_retriever
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader  # Use PyPDF2 for PDF extraction
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()
HUGGINGFACEHUB_API_TOKEN= os.getenv("HUGGINGFACEHUB_API_TOKEN")
system_template = """
You are an AI Assistant that provides answers to users based on the document.
Your knowledge comes from a database containing brief information about that document.

Rules:
- If the user asks something unrelated to the document, inform them that you can only provide information based on the document.
- If the user asks for a summary of the document or what it is about, provide a brief summary based on the document.
- Provide the user with the most relevant information from the database. Do not answer anything that is not in the document or database.
- If the user mentions their name, greet the user by their name in the next message.
- Always maintain the context from the entire conversation. If the user has expressed an intention or request earlier, consider it in your response.
- If the document does not contain the requested information, politely inform the user.

Context: {context} """
model = None
tokenizer = None

def initialize_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')

def generate_flat5_summary(model, text, max_length=500):
    response = model(text, max_length=max_length, min_length=100, do_sample=False)
    return response[0]['summary_text']


@cl.on_chat_start
async def on_chat_start():
    files = None
    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    file_path = file.path
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"  # Join text from all pages
    
    cl.user_session.set("text", text)
    
    try:
        docs = DocumentLoader.load_document(file_path)
        docsearch = configure_retriever(docs)
        cl.user_session.set("docs", docs)

        messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
       ]

        prompt = ChatPromptTemplate.from_messages(messages)
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", model_kwargs={"temperature": 0})

        chain = RetrievalQA.from_chain_type(
            llm, retriever=docsearch, chain_type_kwargs={"prompt": prompt}
        )
        
        cl.user_session.set("chain", chain)
        cl.user_session.set("retriever", docsearch)
        
        logging.info("Chain setup complete.")

        msg.content = f"Processing `{file.name}` done. You can now ask questions or click the button below to summarize the document!"
        await msg.update()

        actions = [cl.Action(name="summarize", value="summarize", description="Summarize the document")]
        await cl.Message(content="Click below to summarize the document:", actions=actions).send()

    except DocumentLoaderException as e:
        msg.content = str(e)
        await msg.update()

    except Exception as e:
        msg.content = "An error occurred while processing the file."
        await msg.update()
        logging.error(f"Error during file processing: {e}")


@cl.action_callback("summarize")
async def summarize_action(action: cl.Action):
    initialize_model()
    docs = cl.user_session.get("docs")
    text = cl.user_session.get("text")
    if not docs:
        await cl.Message(content="No document found to summarize.").send()
        return
    full_text = "\n".join([doc.page_content for doc in docs])

    summary_prompt = f"Summarize the following data in details: always say in this document \n\n{text}"

    inputs = tokenizer(summary_prompt, return_tensors='pt')


    output_ids = model.generate(
    inputs["input_ids"],
    max_new_tokens=512,       # Increase token count for longer summaries
    num_beams=4,              # Use beam search for better results
    length_penalty=1.0,       # Adjust length penalty (default is 1.0, lower values encourage longer output)
    early_stopping=True       # Stops generation early when a complete sentence is formed  
    )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)


    try:
       
       logging.info(f"Summary of document: {output}")
       await cl.Message(content=f"Summary of the document:\n\n{output}").send()

    except Exception as e:
        logging.error(f"Error during document summarization: {e}")
        await cl.Message(content="An error occurred while summarizing the document.").send()


@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    retriever = cl.user_session.get("retriever")

    relevant_docs = retriever.get_relevant_documents(message.content)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    logging.info(f"context docs are :{context}")

    question = message.content
    try:
        result = await chain.arun({"query": question, "context" :  context})

        logging.info(f"Result from message processing: {result}")

        # Check the result structure and extract the answer
        if isinstance(result, dict):
            answer = result.get('result', 'No answer provided.')
        elif isinstance(result, str):
            answer = result
        else:
            answer = 'I am sorry, I can only respond from the available document.'

        await cl.Message(content=answer).send()

    except Exception as e:
        logging.error(f"Error during message processing: {e}")
        await cl.Message(content="An error occurred while processing your request.").send()
