
from langchain_experimental.agents import create_csv_agent, create_pandas_dataframe_agent
import getpass
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import pandas as pd

import os
import logging
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

az_endpoint="https://gpt-main.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    azure_endpoint=az_endpoint,
    api_version="2025-01-01-preview", 
    temperature=0,
    # other params...
)

file_path = "docs\mental_health_recommendations_100k.csv"  

try:
    df = pd.read_csv(file_path)
    logging.info(f"Successfully loaded CSV: {file_path}")
except Exception as e:
    logging.error(f"Error loading CSV file: {e}")

try:
    agent = create_csv_agent(
        llm,
        file_path,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,#AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        allow_dangerous_code=True
    )
    logging.info("Agent initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing agent: {e}")

def analyze_data(query:str)->str:
    """
    Perform advanced analysis on the CSV data using the agent.
    """
    logging.info(f"Pandas Executing query: {query} ")
    
    try:
        response = agent.invoke(query)
        return response["output"]
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return None
    



if __name__ == "__main__":
    # Prompt for external party ID
    # external_party_id = input("Please enter the external party ID: ")

    queries = [
        "What is the fitness have to take by Haiti, Can you suggest some fitness which can be done in future",
        "How is your appetite? Are you eating more or less than usual for Haiti"
    ]

    # Perform analysis for each query with the external party ID
    for query in queries:
        print(f"\nQuery: {query}")
        result = analyze_data(query)#, external_party_id)
        if result:
            print(f"Result: {result}")
        else:
            logging.error(f"No result returned for query: {query}")
            

#####Pandas agent
# import pandas as pd
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_openai import AzureChatOpenAI
# import os
# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model

# # Load environment variables
# load_dotenv()

# # Initialize OpenAI API key
# api_key = os.environ.get("OPENAI_API_KEY")

# api_key = os.getenv("GOOGLE_API_KEY")
# llm = init_chat_model("gemini-1.5-pro", 
#                       model_provider="google_genai", 
#                       api_key=api_key)


# # Step 1: Load the CSV data into a pandas DataFrame
# csv_file_path = "docs/upcoming_transactions.csv"  # Replace with the actual path to your CSV file
# df = pd.read_csv(csv_file_path)

# # Step 2: Set up the OpenAI LLM (Language Model)
# # Replace 'your-openai-api-key' with your actual OpenAI API key
# # Initialize OpenAI Chat Model (GPT-4o Mini)
# # Initialize OpenAI Chat Model (GPT-4o Mini)
# # llm = AzureChatOpenAI(
# #     deployment_name="gpt-4o",
# #     model_name="o3-mini",
# #     azure_endpoint="https://gpt-4-main.openai.azure.com/",
# #     api_key=api_key,
# #     api_version="2024-12-01-preview",
# #     temperature=None  # Explicitly disable temperature
# # )

# # Step 3: Create the pandas agent
# agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# # Step 4: Define a function to handle user queries
# def analyze_data(query):
#     try:
#         # Use the agent to process the query
#         response = agent.run(query)
#         return response
#     except Exception as e:
#         return f"An error occurred: {e}"

# # # #Example usage
# # if __name__ == "__main__":
# #     # Example query: "What is the total amount of processed transactions?"
# #     user_query = input("Enter your query: ")
# #     result = analyze_data(user_query)
# #     print(result)


