import torch
import numpy as np
import pandas as pd
import streamlit as st
from transformers import pipeline
# import sounddevice as sd
import scipy.io.wavfile as wav
import soundfile as sf
import torchaudio
# import boto3
# import zipfile
# from dotenv import load_dotenv
import os
# from tqdm import tqdm
import librosa
import io
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_recorder_streamlit import audio_recorder

# load_dotenv()

st.markdown(
    """
    <h1 style="text-align: center;">
        Speech to Text <span style="color: lightblue;">App</span>
    </h1>
    """,
    unsafe_allow_html=True
)
col1, col2= st.columns(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cwd = os.getcwd()
# model_path = os.path.join('downloads', 'whisper-finetuned9512')
# print(model_path)
pipe = pipeline(task='automatic-speech-recognition',
                model='Humphery7/africanaccented_englishfinetuned',
                chunk_length_s=30,
                stride_length_s=(15, 3),
                device=device)

# token = "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"

# model = WhisperForConditionalGeneration.from_pretrained(model_path)
# processor = WhisperProcessor.from_pretrained(model_path)
# model.push_to_hub('Humphery7/africanaccented_englishfinetuned')
# processor.push_to_hub('Humphery7/africanaccented_englishfinetuned')

samplerate = 16000
MAX_DURATION = 90

# def get_model_s3(bucket, prefix):
#     aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
#     aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
#     region = os.getenv('REGION')
#
#     session = boto3.Session(region_name=region, aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
#     s3 = session.client('s3')
#     cwd = os.getcwd()
#     download_dir = os.path.join(cwd, 'downloads')
#     os.makedirs(download_dir, exist_ok=True)
#     paginator = s3.get_paginator('list_objects_v2')
#     page_iterator = paginator.paginate(Bucket=bucket, Prefix = prefix)
#
#     for page in page_iterator:
#         for num, obj in enumerate(tqdm(page.get('Contents',[]), desc='object iteration')):
#             key = obj['Key']
#             print(key)
#             filename = key.split('.')[0]
#             if filename == '/whisper-finetuned9512':
#                 download_path = os.path.join(download_dir, os.path.basename(key))
#
#                 s3.download_file(bucket, key, download_path)
#                 return



# Function to record audio from the microphone
# def record_audio(duration=10, samplerate=16000)
def transcribe_audio(audio):
    prediction = pipe(audio, batch_size=8)["text"]
    return prediction

with col1:

    st.header(':blue[Record Audio]')
    duration = st.number_input('Enter Recording duration', max_value=90,
                                   min_value=1, step=1,
                                   format = '%d', value=1)
    # if st.button('Start'):
        # audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        # audio = audio.reshape(-1)
        # sd.wait()
    audio_bytes = audio_recorder()
    # print(audio_bytes)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        # st.write('Click image to record')
        # audio_bytes = audio_recorder(text="",
        #                              recording_color="#e8b62c",
        #                              neutral_color="#6aa36f",
        #                            icon_name="user",icon_size="6x", pause_threshold=duration, energy_threshold=0)
        # print(audio_bytes)
        # if audio_bytes:
        #     st.write('Recording')
        #     st.audio(audio_bytes, format="audio/wav")
        #
        try:
            audio_stream = io.BytesIO(audio_bytes)
            audio, sample_rate = librosa.load(audio_stream, sr=None, mono=True)

            # st.write(f"Audio length: {len(audio)} samples")
            # st.write(f"Sample rate: {sample_rate} Hz")

            result = transcribe_audio(audio)
            st.write(':blue[Transcription:]')
            st.write(result)

        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

with col2:
    st.header(':blue[Upload Audio]')
    uploaded_audio = st.file_uploader('Upload an Audio File', type=['wav', 'mp3'])

    if uploaded_audio is not None:
        try:

            audio_bytes = uploaded_audio.read()
            audio, sr = librosa.load(io.BytesIO(audio_bytes),sr=None, mono=True)
            duration_in_seconds = librosa.get_duration(y=audio, sr=sr)

            if duration_in_seconds > MAX_DURATION:
                st.error(f"Uploaded audio is too long! Max allowed duration is {MAX_DURATION} seconds.")
            else:
                st.success("Audio file uploaded successfully!")
                st.audio(uploaded_audio)
                if st.button('Transcribe'):
                    result = transcribe_audio(audio)
                    st.write(":blue[Transcription:]")
                    st.write(result)
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")


