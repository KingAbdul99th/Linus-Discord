
import asyncio
from typing import Any

import discord

from models.linus_model import LinusModel


XTTS_SAMPLE_RATE = 24000


class LinusClient(discord.Client):
    def __init__(self, *, intents: discord.Intents, **options: Any) -> None:
        super().__init__(intents=intents, **options)
        self.model = LinusModel(enable_voice=True)
        self.voice_client = None

    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        self.bot_user = discord.utils.get(self.users, name="Linus")

    async def on_message(self, message: discord.message):
        print(f'Message from {message.author} on {message.guild}: {message.content}')

        if message.guild.name != "Testbench":
            print("not testbench server")
            return
        
        if message.content.startswith(f"<@{self.bot_user.id}> /connectvoice"):
            voice_channel_name = message.clean_content.split(" ")[-1]
            await self.initialize_voice_client(voice_channel_name)
            return

        if message.content.startswith(f"<@{self.bot_user.id}> /disconnectvoice"):
            await self.voice_client.disconnect()
            self.voice_client = None
            return

        if message.content.startswith(f"<@{self.bot_user.id}>"):
            print(message.clean_content)
            response = await self.model.respond(message)

            await message.channel.send(response)

            if self.voice_client:
                await self.play_response_stream(response)
            return

    async def initialize_voice_client(self, voice_channel_name):
        self.voice_channel = discord.utils.get(self.get_all_channels(), name=voice_channel_name)
        self.voice_client = await self.voice_channel.connect(timeout=300)
        print(f"connected to {self.voice_client.channel}")

    async def play_response_stream(self, text: str):
        print("playing voice")

        stream_reader = self.model.generate_voice_stream(text)
        ffmpeg_source = discord.FFmpegOpusAudio(
            stream_reader,  # type: ignore # Pass our RawIOBase reader here (duck typing works)
            pipe=True,
            before_options=f"-f s16le -ac 1 -ar {XTTS_SAMPLE_RATE}",
            options="-loglevel warning -vn",
        )
        
        def after_play(error):
            if error:
                print(f"Error during TTS playback: {error}")
            print("Playback finished or stopped. Closing TTS reader.")
            if stream_reader:
                stream_reader.close()

        self.voice_client.play(ffmpeg_source, after=after_play)
        print(
            "Started voice_client.play() with FFmpegOpusAudio piping TTSStreamReader."  # Keep essential log
        )

        print("Done playing voice")

    async def play_response(self, response):
        print("playing voice")

        audio = self.model.generate_voice(response)
        self.voice_client.play(discord.FFmpegPCMAudio(source=audio))
        self.voice_client.source = discord.PCMVolumeTransformer(self.voice_client.source)
        self.voice_client.source.volume = 1

        while self.voice_client.is_playing():
            await asyncio.sleep(1)
        print("Done playing voice")
