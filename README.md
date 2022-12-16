# discord-yumi-bot

This discord bot will detect if the pictures in a message from yumi or not

## How to run

- Make sure you already have `model.tflite` file, if not see `How to train model` section

- Install Tensorflow (https://www.tensorflow.org/install/)

- Install `dotenv` and `discord.py` library

```bash
pip install python-dotenv
pip install discord.py
```

- Create `.env` file with `DISCORD_TOKEN` and `DISCORD_KEYWORD` variable

```bash
DISCORD_TOKEN=token_here
DISCORD_KEYWORD=!yumi
```

`DISCORD_KEYWORD` is the keyword that will trigger the bot to reply

- Run `main.py` script

```bash
python main.py
```

## How to run in Raspberry Pi

- Make sure you already have `model.tflite` file, if not see `How to train model` section

- Install Tensorflow Lite Runtime (https://www.tensorflow.org/lite/guide/python)

```bash
pip install tflite-runtime
```

- Install `dotenv` and `discord.py` library

```bash
pip install python-dotenv
pip install discord.py
```

- Create `.env` file with `DISCORD_TOKEN` and `DISCORD_KEYWORD` variable

```bash
DISCORD_TOKEN=token_here
DISCORD_KEYWORD=!yumi
```

`DISCORD_KEYWORD` is the keyword that will trigger the bot to reply

- Run `main_tflite.py` script

```bash
python main_tflite.py
```

## How to train model

**Model training can only be done on PC!**

- Download and extract dataset into `yumi_dataset` folder (https://onedrive.live.com/download?cid=E5A63D34BCCB21B3&resid=E5A63D34BCCB21B3%2126891&authkey=AGj-KR9VqDLd23s)

- Run `train_model.py` script

```bash
python train_model.py
```

- Run `convert_to_tflite.py` script to get `model.tflite` file

```bash
python convert_to_tflite.py
```
