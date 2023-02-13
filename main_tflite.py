import os

import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import urllib.request
import mimetypes
import discord
from dotenv import load_dotenv

# Get discord bot parameters
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
KEYWORD = os.getenv('DISCORD_KEYWORD')

# Start discord bot
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Load model
model_path = "model.tflite"
model = tflite.Interpreter(model_path=model_path)
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
print('Model loaded.')

def is_url_image(url):    
    mimetype, encoding = mimetypes.guess_type(url)

    return (mimetype and mimetype.startswith('image'))

def get_image_from_url(image_url):
    req = urllib.request.Request(
        image_url, 
        data=None, 
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
    )

    req = urllib.request.urlopen(req)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)

    return img

def predict_image(model, image_url):
    image_size = (224, 224)
    frame = get_image_from_url(image_url)
    frame = cv2.resize(frame, image_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.array(frame, dtype=np.float32)
    frame = np.expand_dims(frame, axis=0)

    # Predict
    model.set_tensor(input_details[0]['index'], frame)
    model.invoke()

    # Process mask
    pred_output = model.get_tensor(output_details[0]['index'])

    if pred_output[0] < 0.5:
        confidence = 1 - pred_output[0][0]
        output_class = 'notyumi'
    else:
        confidence = pred_output[0][0]
        output_class = 'yumi'

    # print('Image prediction:', output_class)
    # print('Confidence:', confidence)

    return output_class, confidence

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')

@client.event
async def on_message(message):
    # Exception to prevent bot from replying to itself
    if message.author == client.user:
        return
    
    if message.content == KEYWORD:
        if (len(message.attachments) != 0) and (is_url_image(message.attachments[0].url)):
            output_class, confidence = predict_image(model, str(message.attachments[0].url))
            response = 'Yumi the best' + '\n' + 'Image prediction: ' + str(output_class) + '\n' + 'Confidence: ' + str(confidence)
            await message.channel.send(response, reference=message)

if __name__ == '__main__':
    client.run(TOKEN)
