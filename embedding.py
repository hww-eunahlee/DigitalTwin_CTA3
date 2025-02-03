import os
from openai import AzureOpenAI

# Creates a ChatGPT client using Azure OpenAI credentials
client = AzureOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),  # Retrieves Azure OpenAI endpoint
    api_key=os.getenv('AZURE_OPENAI_KEY'),  # Retrieves API key
    api_version="2023-12-01-preview",  # Specifies API version
)

#########################################################
# STEP 1. Read text file and store non-empty lines in a DataFrame 
#########################################################

def create_index(filename):
    df = pd.DataFrame(columns=["text"])  # Initialize an empty DataFrame
    file = filename + ".txt"  # Append ".txt" to filename

    with open(file, "r", errors="ignore") as f:
        text = f.read().split("\n")  # Read the file and split into lines

    for line in text:
        if line.strip():  # If line is not empty empty lines
            df = pd.concat([df, pd.DataFrame([{"text": line}])], ignore_index=True)  # Add lines to DataFrame
