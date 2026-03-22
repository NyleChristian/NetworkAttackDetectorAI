from email import header

from google.adk.agents.llm_agent import Agent
import os
from dotenv import load_dotenv
from google.adk.tools.google_search_tool import GoogleSearchTool
import joblib
import joblib
from common import *
from functions import *
import io
import pandas as pd
import csv

load_dotenv()

def read_csv_to_list(filename):
    data = []
    # Open the file using a context manager for proper handling
    with open(filename, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data.append(row)
    return data


def preprocess_input(attack_type,input_data):
    # Implement your preprocessing logic here

    with open(f'headers_only_{attack_type}.csv', mode='r', newline='', encoding='utf-8') as csv_file:
        header_reader = csv_file.readline().strip()  # Read the first line (header) and remove any leading/trailing whitespace

   
    input_data_with_header = header_reader + "\n" + input_data

    file_like_object = io.StringIO(input_data_with_header)
    df = pd.read_csv(file_like_object)

    return df


google_search = GoogleSearchTool()
dos_model = joblib.load('DOS_model.joblib')
probe_model = joblib.load('PROBE_model.joblib')
r2l_model = joblib.load('R2L_model.joblib')
u2r_model = joblib.load('U2R_model.joblib')

def convert_prediction_to_string(prediction):
    """
    Convert numpy.ndarray prediction to string label
    
    Args:
        prediction: numpy.ndarray from model
        
    Returns:
        str: "match" or "not match"
    """
    # Get the predicted class
    predicted_value = int(prediction[0]) if isinstance(prediction, np.ndarray) else int(prediction)
    
    return "match" if predicted_value == 1 else "not match"

def dos_predict(input_data):
    predict = dos_model.predict(preprocess_input("DOS", input_data))
    return convert_prediction_to_string(predict)
  

def probe_predict(input_data):
    predict = probe_model.predict(preprocess_input("PROBE", input_data))
    return convert_prediction_to_string(predict)
def r2l_predict(input_data):
    predict = r2l_model.predict(preprocess_input("R2L", input_data))
    return convert_prediction_to_string(predict)

def u2r_predict(input_data):
    predict = u2r_model.predict(preprocess_input("U2R", input_data))
    return convert_prediction_to_string(predict)

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='You are a network security agent tasked with classifying network traffic as either normal or one of 4 attack types: DOS, PROBE, R2L, U2R.',
    instruction="""
    If a file is attached, you MUST call the tool for each row, and make a prediction on each row individually (skip the first row).
    Only call the tool that corresponds to the attack type the user wants to test for. 
    User needs to provide the attack type they want to test for and the network traffic data in csv format. 
    Input data MUST be passed as a String.
    Output should be in a tabular format.
    """,
    tools=[dos_predict, probe_predict, r2l_predict, u2r_predict],
)
