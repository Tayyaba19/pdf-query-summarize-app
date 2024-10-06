import os
import chainlit as cl
from dotenv import load_dotenv
import logging
import warnings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,)
from langchain.chains import LLMChain,RetrievalQA
from documentLoader.file import DocumentLoader, DocumentLoaderException, configure_retriever


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*The class `LLMChain` was deprecated.*")

load_dotenv()
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

system_template = """You are a AI Assistant that provides answere to user based on the document.
        Your knowledge comes from a database containing bief infomration about that document.

    Rules:
    - If User ask something other then database. just say you are AI assistant you only provide information about document.
    - If user ask about the summery of document or ask what is this document about. provide the brief summery of the document.
    - Provide the user with the most relevant info from the database. Dont answare anything that is not in database.
    - If user tell their name. then greet the user in first message. 
    - Always maintain context from the entire conversation when responding. If the user has expressed an intention or made a request in a previous message, follow through on that in your response.


    Context: {context} """


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

    try:
        docs = DocumentLoader.load_document(file_path)
        docsearch = configure_retriever(docs)
        cl.user_session.set("docs", docs) 

        messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),]
        prompt = ChatPromptTemplate.from_messages(messages)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

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
    docs = cl.user_session.get("docs")
    if not docs:
        await cl.Message(content="No document found to summarize.").send()
        return

    # Summarize the document using the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Combine all pages into a single document string
    full_text = "\n".join([doc.page_content for doc in docs])

    # Create the prompt for summarization
    summary_prompt = f"Summarize the following document:\n\n{full_text}"

    # Chain setup to run the LLM
    summary_chain = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(summary_prompt)
    )

    try:
        # Get the summary using the chain
        summary = await summary_chain.arun({})

        logging.info(f"Summary of document: {summary}")

        # Send the summary back to the user
        await cl.Message(content=f"Summary of the document:\n\n{summary}").send()

    except Exception as e:
        logging.error(f"Error during document summarization: {e}")
        await cl.Message(content="An error occurred while summarizing the document.").send()

@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    retriever = cl.user_session.get("retriever")

    # Retrieve relevant documents from retriever
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
