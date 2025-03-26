import openai
import azure.cognitiveservices.speech as speechsdk
import requests
import json

# Azure Credentials
AZURE_OPENAI_KEY = "your-azure-openai-key"
AZURE_OPENAI_ENDPOINT = "your-azure-openai-endpoint"
AZURE_SPEECH_KEY = "your-azure-speech-key"
AZURE_SPEECH_REGION = "your-speech-region"
AZURE_TRANSLATOR_KEY = "your-translator-key"
AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com/"

# Initialize OpenAI Client
openai.api_key = AZURE_OPENAI_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT

# Function to Convert Speech to Text
def speech_to_text():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    
    print("Listening...")
    result = speech_recognizer.recognize_once()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "Sorry, I didn't catch that."
    else:
        return "Speech recognition failed."

# Function to Translate Text
def translate_text(text, target_language):
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_TRANSLATOR_KEY,
        'Ocp-Apim-Subscription-Region': AZURE_SPEECH_REGION,
        'Content-Type': 'application/json'
    }
    body = [{'text': text}]
    params = {'api-version': '3.0', 'to': target_language}

    response = requests.post(AZURE_TRANSLATOR_ENDPOINT + "translate", headers=headers, params=params, json=body)
    result = response.json()
    
    return result[0]['translations'][0]['text']

# Function to Generate Response from Azure OpenAI
def generate_response(prompt, language="en"):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a multilingual AI assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Function to Convert Text to Speech
def text_to_speech(text, language="en-US"):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name = f"{language}-Neural"
    
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text_async(text).get()

# Main Function
def multilingual_assistant():
    print("Speak now...")
    user_input = speech_to_text()
    print(f"Recognized: {user_input}")
    
    if user_input.lower() == "exit":
        return
    
    target_language = input("Enter target language (e.g., fr for French, es for Spanish, de for German): ")
    
    translated_input = translate_text(user_input, "en")  # Convert to English
    ai_response = generate_response(translated_input)
    translated_response = translate_text(ai_response, target_language)  # Convert back to target language
    
    print(f"AI Response: {translated_response}")
    
    text_to_speech(translated_response, target_language)

# Run the Assistant
if __name__ == "__main__":
    multilingual_assistant()
