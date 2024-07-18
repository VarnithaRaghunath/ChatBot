import datetime
import json
import os
import random
import subprocess
import sys

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clean_response(response):
    """
    Clean up the response by removing HTML tags and attributes.
    """
    cleaned_response = response.replace('<a', '').replace('href=', '').replace('target="_blank"', '').replace('</a>', '').replace('>', ' ')
    return cleaned_response


def open_new_word():
    if sys.platform == 'win32':  # Check if the system is Windows
        try:
            os.startfile("winword")  # Open Microsoft Word using the default application
        except OSError:
            print("Failed to open Microsoft Word.")
    else:
        print("Opening Microsoft Word is only supported on Windows.")

def open_chrome():
    if sys.platform == 'win32':  # Check if the system is Windows
        try:
            subprocess.Popen(['start', 'chrome'], shell=True)  # Open Google Chrome
        except OSError:
            print("Failed to open Google Chrome.")
    elif sys.platform == 'darwin':  # Check if the system is macOS
        try:
            subprocess.Popen(['open', '-a', 'Google Chrome'])  # Open Google Chrome
        except OSError:
            print("Failed to open Google Chrome.")
    else:
        print("Opening Google Chrome is only supported on Windows and macOS.")
        
def get_weather():
    weather_conditions = [
        {"weather": "Sunny", "temperature": "32°C"},
        {"weather": "Partly cloudy", "temperature": "28°C"},
        {"weather": "Cloudy", "temperature": "26°C"},
        {"weather": "Rainy", "temperature": "24°C"},
        {"weather": "Thunderstorms", "temperature": "22°C"},
    ]
    weather_data = random.choice(weather_conditions)
    return f"The weather in Bangalore today is {weather_data['weather']} with a temperature of {weather_data['temperature']}."


with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Chatify"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                # Check if the response contains HTML tags
                if "<" in response and ">" in response:
                    cleaned_response = clean_response(response)
                    print(f"{bot_name}: {cleaned_response}")
                elif intent['tag'] == 'datetime':
                    now = datetime.datetime.now()
                    current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
                    response = random.choice(intent['responses']).replace("{{datetime}}", current_datetime)
                    print(f"{bot_name}: {response}")
                elif intent['tag'] == 'open_notepad':
                    subprocess.Popen('notepad.exe')  # Open Notepad using subprocess module
                    print(random.choice(intent['responses']))
                elif intent['tag'] == 'open_new_word':
                    open_new_word()  # Call a function to open a new instance of Word
                    print("I have opened a new instance of Microsoft Word for you.")
                elif intent['tag'] == 'open_chrome':
                    open_chrome()
                    print("I have opened Google Chrome for you.")
                elif intent['tag'] == 'get_weather':
                    print(get_weather())
                else:
                    print(f"{bot_name}: {response}")
    else:
        print(f"{bot_name}: I do not understand...")
        
