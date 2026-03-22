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
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Ensure ADK treats this as a chat model and knows where Ollama lives
ollama_llm = LiteLlm(
    model=f"ollama_chat/{OLLAMA_MODEL}",
    api_base=OLLAMA_API_BASE,
)

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

root_agentold = Agent(
    model=ollama_llm,
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

dos_Agent = Agent(
    model='gemini-2.5-flash',
    name='dos_Agent',
    description='You are a network security agent tasked with classifying network traffic as either normal or  DOS.',
    instruction="""
    Use the dos_predict tool to classify the input network traffic data as either normal or DOS attack.
    If a file is attached, you MUST call the tool for each row, and make a prediction on each row individually (skip the first row).
    Only call the tool that corresponds to the attack type the user wants to test for. 
    User needs to provide the attack type they want to test for and the network traffic data in csv format. 
    Input data MUST be passed as a String.
    Do not give output to user. only give output to the next AI Agent.
    """,
    tools=[dos_predict],
)

probe_Agent = Agent(
    model='gemini-2.5-flash', 
    name='probe_Agent',
    description='You are a network security agent tasked with classifying network traffic as either normal or PROBE.',
    instruction="""   Use the probe_predict tool to classify the input network traffic data as either normal or PROBE attack.   
    If a file is attached, you MUST call the tool for each row, and make a prediction on each row individually (skip the first row).
    Only call the tool that corresponds to the attack type the user wants to test for.          
    User needs to provide the attack type they want to test for and the network traffic data in csv format.
    Input data MUST be passed as a String.
    Do not give output to user. only give output to the next AI Agent.
    """,
    tools=[probe_predict],
)

r2l_Agent = Agent(
    model='gemini-2.5-flash',   

    name='r2l_Agent',
    description='You are a network security agent tasked with classifying network traffic as either normal or R2L.',
    instruction="""   Use the r2l_predict tool to classify the input network traffic data as either normal or R2L attack.
    If a file is attached, you MUST call the tool for each row, and make a prediction on each row individually (skip the first row).
    Only call the tool that corresponds to the attack type the user wants to test for.
    User needs to provide the attack type they want to test for and the network traffic data in csv format.
    Input data MUST be passed as a String.
    Do not give output to user. only give output to the next AI Agent.
    """,
    tools=[r2l_predict],

)


u2r_Agent = Agent(
    model='gemini-2.5-flash',
    name='u2r_Agent',
    description='You are a network security agent tasked with classifying network traffic as either normal or U2R.',
    instruction="""   Use the u2r_predict tool to classify the input network traffic data as either normal or U2R attack.
    If a file is attached, you MUST call the tool for each row, and make a prediction on each row individually (skip the first row).
    Only call the tool that corresponds to the attack type the user wants to test for.
    User needs to provide the attack type they want to test for and the network traffic data in csv format.
    Input data MUST be passed as a String.
    Do not give output to user. only give output to the next AI Agent.  
    """,
    tools=[u2r_predict],

)




parallel_research_agent = ParallelAgent(
     name="Parallel_network__attack_research_agent",
     sub_agents=[dos_Agent, probe_Agent],
     description="Runs multiple research agents in parallel to gather information."
 )

merger_agent = LlmAgent(
    name="MergerAgent",
    model="gemini-2.5-flash",
    description="Merges the results from parallel research agents and synthesizes the information.",
    instruction="""
    You are a merger agent that takes the outputs of multiple parallel research agents and synthesizes the information into a cohesive format.
    Your task is to take the predictions from each agent and compile them into a single report that indicates which attack types were detected in the input network traffic data.
    The output should be in a clear, tabular format that lists each row number of input data along with the corresponding predictions from each agent.
    Do not include input data itself in the output, only include the row number and the predictions from each agent.
   

      """,
)

sequential_pipeline_agent = SequentialAgent(
     name="Network_Attack_Classification_Agent",
     # Run parallel research first, then merge
     sub_agents=[parallel_research_agent, merger_agent],
     description="Coordinates parallel agents and synthesizes the results."
 )


root_agent  =sequential_pipeline_agent