import pdfplumber
import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain_community.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

# Define LLM setup
class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": {**model_kwargs}})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_content = output.read()
        try:
            model_predictions = json.loads(response_content)
            generated_text = model_predictions.get("generated_text", "")
            return generated_text
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON response: {e}")

access_key_id = os.getenv('ACCESS_KEY')
secret_access_key = os.getenv('SECRET_KEY')

config = {
    "region_name": "us-east-1",
    "endpoint_name": "jumpstart-dft-meta-textgeneration-l-20240423-131037",
    "model_kwargs": {
        "max_new_tokens": 400,
        "top_p": 0.9,
        "stop": None,
        "temperature": 0.7
    }
}
region_name = config['region_name']
endpoint_name = config['endpoint_name']
model_kwargs = config['model_kwargs']

session = boto3.Session(region_name=region_name, aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
client = session.client("runtime.sagemaker")

content_handler = ContentHandler()

llm = SagemakerEndpoint(
    endpoint_name=endpoint_name,
    region_name=region_name,
    client=client,
    model_kwargs=model_kwargs,
    content_handler=content_handler
)

prompt_template = """
<s>[INST] You are an expert assistant specialized in the EU AI Act. Your goal is to provide accurate, clear, and engaging information. Ensure your responses are concise and relevant to the user's query. If the user's question is not related to the EU AI Act, politely redirect them back to the topic without sounding robotic.

User Query: {input} 
Context: {context} 

Provide a detailed and informative answer based on the context above.
[/INST]
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["input", "context"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt, output_key="out_gen")

# Functions for extracting, cleaning, and chunking text and tables
def extract_text_from_pdf(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i in range(min(97, total_pages)):
            page = pdf.pages[i]
            text = page.extract_text()
            if text:
                extracted_text.append((i + 1, text.strip()))
    return extracted_text

def extract_tables_from_pdf(pdf_path):
    extracted_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i in range(97, min(108, total_pages)):
            page = pdf.pages[i]
            tables = page.extract_tables()
            for table in tables:
                if table:
                    extracted_tables.append((i + 1, table))
    return extracted_tables

def clean_extracted_text(extracted_text):
    cleaned_text = []
    for page_num, text in extracted_text:
        cleaned_text_content = text.replace('\n', ' ').strip()
        cleaned_text.append((page_num, cleaned_text_content))
    return cleaned_text

def clean_extracted_table(table):
    cleaned_table = []
    for row in table:
        cleaned_row = [cell.strip() if cell else "" for cell in row]
        if any(cleaned_row) and not all(cell in ["", "\uf09f"] for cell in cleaned_row):
            cleaned_table.append(cleaned_row)
    return cleaned_table

def clean_extracted_tables(extracted_tables):
    cleaned_tables = []
    for page_num, table in extracted_tables:
        cleaned_table = clean_extracted_table(table)
        cleaned_tables.append((page_num, cleaned_table))
    return cleaned_tables

def chunk_text_into_documents(text, chunk_size=256, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([text])
    return docs

def process_pdf(pdf_path):
    # Extract and clean text
    extracted_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_extracted_text(extracted_text)

    # Extract and clean tables
    extracted_tables = extract_tables_from_pdf(pdf_path)
    cleaned_tables = clean_extracted_tables(extracted_tables)

    # Combine cleaned text into a single string
    all_cleaned_text = " ".join(text for _, text in cleaned_text)

    # Chunking only text since tables are an unstructured data form
    docs = chunk_text_into_documents(all_cleaned_text)
    return docs

# Step 1: Process the PDF and Save Embeddings to ChromaDB
pdf_path = 'pdf2.pdf'
db_path = 'db'
embedding = FastEmbedEmbeddings()

def initialize_retrieval_system(pdf_path, db_path, embedding):
    """
    Initialize the retrieval system with ChromaDB.
    """
    # Process the PDF
    docs = process_pdf(pdf_path)
    # Save embedded documents to ChromaDB
    vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=db_path
    )
    print("Documents processed and embeddings saved to ChromaDB.")
    return vector_store

vector_store = initialize_retrieval_system(pdf_path, db_path, embedding)

def get_answer_from_query(query, vector_store, llm_chain):
    """
    Perform retrieval and generate an answer for a given query.
    Returns:
        str: The generated answer.
    """
    # Perform retrieval
    docs = vector_store.similarity_search(query, k=5)
    # Combine the relevant documents into a single context
    context = " ".join([doc.page_content for doc in docs])
    # Generate answer using the LLM chain
    # Create the formatted input for the LLM
    #formatted_input = prompt.format(input=query, context=context)

    # Step 3: Generate Response Using LLM
    response = llm_chain.invoke({"input": query, "context": context})
    
    return response['out_gen'], context

def main_pipeline(query):
    return get_answer_from_query(query, vector_store, llm_chain)
