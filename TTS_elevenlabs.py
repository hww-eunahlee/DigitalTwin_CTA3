import os, elevenlabs
from pythonosc import udp_client


def get_speech(answer):
    elevenlabs.set_api_key(os.environ.get("ELEVENLABS_KEY"))
    audio = elevenlabs.generate(
        text=answer,
        # need a voice id
        voice="Mvi9cp0TjLS2CVndrxqi",
        model="eleven_multilingual_v2",
    )
    #elevenlabs.play(audio),
    file_path = os.path.join("voices/", "audio.wav")

    with open(file_path, "wb") as f:
        f.write(audio)
    pass
