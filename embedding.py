import pandas as pd
import numpy as np
import faiss, glob, pickle, os
#from config import *
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


    ######################################################
    # STEP 2. Generate embeddings for each line in batches
    ######################################################

    embeddings = []
    BATCH_SIZE = 4  # Set batch size for embedding requests

    for i in range(0, len(df), BATCH_SIZE):
        print(df["text"].iloc[i : i + BATCH_SIZE])  # Print batch content for reference
        
        # Display progress
        if (i + BATCH_SIZE) < len(df):
            print(f"PROGRESS: Getting indices {i} to {i + BATCH_SIZE}, out of {len(df)}")
        else:
            print(f"PROGRESS: Getting indices {i} to {len(df)}, out of {len(df)}")

        batch_text = df["text"].iloc[i : i + BATCH_SIZE].tolist()  # In DataFrame, in the location from i to i+4, extract the text and make it into a list
        batch_embeddings = get_embedding_batch(batch_text)  # Get embeddings for batch
        embeddings.extend(batch_embeddings)  # Append to embeddings list


    #########################################################
    # STEP 3. Create and save FAISS index 
    #########################################################
    df["embedding"] = embeddings  # Store embeddings in DataFrame
    embeddings = np.vstack(df["embedding"].values).astype("float32")  # Convert embeddings to NumPy array
    dimension = embeddings.shape[1]  # Get embedding dimension
    index = faiss.IndexFlatL2(dimension)  # Create FAISS index (L2 distance)
    index.add(embeddings)  # Add embeddings to FAISS index

    faiss.write_index(index, "knowledge/knowledge_index.txt")  # Save FAISS index to file

    #########################################################
    # STEP 4. Save text data for later retrieval
    #########################################################
    with open("knowledge/knowledge.pkl", "wb") as f:
        pickle.dump(df["text"], f)  # Save original text as a pickle file


def get_embedding_batch(text_list):
    # Calls OpenAI's embedding model and returns embeddings
    response = client.embeddings.create(model="text-embedding-ada-002", input=text_list)
    return [data.embedding for data in response.data]  # Extract embeddings # https://platform.openai.com/docs/guides/embeddings
    # look into every block of data in response. For every block of data, go to data, embedding section and return that


# Run the function to create an index from "knowledge/context.txt"
create_index("knowledge/context")
