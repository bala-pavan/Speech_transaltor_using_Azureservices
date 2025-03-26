import azure.cognitiveservices.speech as speechsdk
from googletrans import Translator
import asyncio
from langchain_openai import AzureChatOpenAI

# Initialize Azure Speech Services
AZURE_SPEECH_KEY = "DsEGOLn6OLwXnASP7c5K8qr7dQoRxbGJNLrXyFvihP6ODosIVfSeJQQJ99BCACL93NaXJ3w3AAAYACOGKPar"
AZURE_REGION = "australiaeast"


az_endpoint="https://gpt-main.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"


# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    azure_endpoint=az_endpoint,
    api_version="2025-01-01-preview", 
    temperature=0,
    # other params...
)

async def speech_to_text():
    """Convert speech to text using Azure Speech Services."""
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    
    print("Listening...")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, recognizer.recognize_once)
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: ", result.text)
        return result.text
    else:
        print("Speech not recognized.")
        return ""

async def translate_text(text, dest_language="en"):
    """Translate text to the desired language using Google Translator."""
    translator = Translator()
    translated = await translator.translate(text, dest=dest_language) 
    return translated.text, translated.src

async def get_ai_response(prompt, language="en"):
    """Get a response from Azure OpenAI's GPT model."""
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        llm.invoke,
        [{"role": "user", "content": prompt}],
    )
    return response.content

async def text_to_speech(text, language="en"):
    """Convert text to speech using Azure Speech Services."""
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, synthesizer.speak_text_async, text)

async def multilingual_ai_assistant():
    """Run the multilingual AI assistant."""
    #Get user input via speech
    user_input = await speech_to_text()
    if not user_input:
        return
    
    #Translate user input to English
    translated_text, detected_lang = await translate_text(user_input, "en")
    print(f"Translated to English: {translated_text} (Detected Language: {detected_lang})")
    
    # Get AI response in English
    ai_response = await get_ai_response(translated_text)
    print(f"AI Response in English: {ai_response}")
    
    # Translate AI response back to the detected language
    translated_response, _ = await translate_text(ai_response, detected_lang)
    print(f"Translated Response: {translated_response}")
    
    # Convert translated response to speech
    await text_to_speech(translated_response, detected_lang)

if __name__ == "__main__":
    asyncio.run(multilingual_ai_assistant())