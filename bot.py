
import asyncio

from typing import Any
import discord
import io
import wave

from discord import opus

from models.linus_model import LinusModel


class LinusClient(discord.Client):
    def __init__(self, *, intents: discord.Intents, **options: Any) -> None:
        super().__init__(intents=intents, **options)
        self.model = LinusModel()
        self.encoder = opus.Encoder()
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
                await self.play_response(response)
            return

    async def initialize_voice_client(self, voice_channel_name):
        self.voice_channel = discord.utils.get(self.get_all_channels(), name=voice_channel_name)
        self.voice_client = await self.voice_channel.connect(timeout=300)
        print(f"connected to {self.voice_client.channel}")

    async def play_response_stream(self, text: str):
        print("playing voice")

        audio = self.model.generate_voice_stream(text)
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as vfout:
            vfout.setnchannels(1)
            vfout.setsampwidth(2)
            vfout.setframerate(24000)
            vfout.writeframes(b"")
        audio_buffer.seek(0)

        for i, chunk in enumerate(audio):
            print(f"processing audio chunk {i}")
            audio_buffer.write(chunk)
            # encoded_bytes = self.encoder.encode(chunk, self.encoder.SAMPLES_PER_FRAME)
            # print(encoded_bytes)
            stream = await discord.FFmpegPCMAudio(chunk)
            self.voice_client.send_audio_packet(stream, encode=False)
            # if not self.voice_client.is_playing():
            #     source = discord.FFmpegPCMAudio(source=audio_buffer, pipe=True)
            #     volume_source = discord.PCMVolumeTransformer(source, volume=1)
            #     self.voice_client.play(volume_source)

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
