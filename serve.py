
import os
import asyncio

from dotenv import load_dotenv
import discord

from utils.bot_logger import initialize_discord_client_logger
from bot import LinusClient

load_dotenv()


def main():
    initialize_discord_client_logger()
    intents = discord.Intents.default()
    intents.typing = False
    intents.members = True
    intents.presences = False
    intents.message_content = True
    client = LinusClient(intents=intents, auto_connect_voice=True)

    asyncio.run(client.start(os.getenv('DISCORD_BOT_TOKEN'), reconnect=True))

if __name__ == "__main__":
    main()
