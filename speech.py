import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import azure.cognitiveservices.speech as speechsdk
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

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


# Azure Speech Services credentials
SPEECH_KEY = "DsEGOLn6OLwXnASP7c5K8qr7dQoRxbGJNLrXyFvihP6ODosIVfSeJQQJ99BCACL93NaXJ3w3AAAYACOGKPar"
SERVICE_REGION = "australiaeast"

def speech_to_text():
    # Initialize the speech recognizer
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print("Say something...")
    result = speech_recognizer.recognize_once()

    # Check result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"Recognized: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized.")
        return None
        
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech recognition canceled: {cancellation_details.reason}")
        return None
    


def text_to_speech(text):
    # Initialize speech configuration
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
    
    # Set up the speech synthesizer
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    # Convert text to speech
    result = synthesizer.speak_text_async(text).get()

    # Check for errors
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesis successful!")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation.error_details}")


def chatbot_conversation():
    """Run a conversational chatbot using Azure Speech and LLM."""
    print("Starting the chatbot. Say 'exit' to quit.")
    while True:
        # Step 1: Get user input via speech
        user_input = speech_to_text()

        if not user_input:
            print("I didn't get the response, Please try again.")
            continue

        # Exit condition
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break

        
        #translated_text, detected_text=translate_text(user_input,"en")
        #ai_response = get_ai_response(translated_text)
        #translated_response, _ = translate_text(ai_response, detected_lang)

        
        # Step 2: Generate response using LLM
        response = llm.invoke([{"role": "user", "content": user_input}])
        chatbot_response = response.content
        print(f"Chatbot: {chatbot_response}")

        # Step 3: Convert chatbot response to speech
        text_to_speech(chatbot_response)


# Start the chatbot
chatbot_conversation()


# speech_to_text()
# text_to_speech("Hello, this is Azure Text-to-Speech in action!")

