import torch
import numpy as np
import pandas as pd
import streamlit as st
from transformers import pipeline
import sounddevice as sd
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
cwd = os.getcwd()
model_path = os.path.join('downloads', 'whisper-finetuned9512')
print(model_path)
pipe = pipeline(task='automatic-speech-recognition',
                model=model_path,
                chunk_length_s=30,
                stride_length_s=(15, 3),
                device=device)

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
    if st.button('Start Recording'):
        st.write('Recording')
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        audio = audio.reshape(-1)
        sd.wait()
        # output_file = "recording.wav"
        # sf.write(output_file, audio, samplerate)
        st.success("Recording Success")
        # st.audio(output_file)
        result = transcribe_audio(audio)
        st.write(f'Transcribed audio: {result}')

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


