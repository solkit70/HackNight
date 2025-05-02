import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import langdetect
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_username = os.getenv('NEO4J_USERNAME')
neo4j_password = os.getenv('NEO4J_PASSWORD')

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize Neo4j driver as a session state variable
if 'neo4j_driver' not in st.session_state:
    st.session_state.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

# Streamlit UI
st.title("Immigrant-Friendly AI Translator with Neo4j Memory")

# Record voice and transcribe
st.header("Voice Transcription and Translation")
if st.button("Start Recording"):
    with st.spinner("Recording... Please speak into the microphone."):
        fs = 44100  # Sample rate
        duration = 10  # Duration in seconds
        st.write("Recording for 10 seconds...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished

        # Save the recording to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, fs, recording)

        with st.spinner("Transcribing..."):
            with open(temp_file.name, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                transcript_text = response.text
                st.write("**Transcription:**")
                st.write(transcript_text)

        # Detect language
        detected_lang = langdetect.detect(transcript_text)
        st.write(f"**Detected Language:** {detected_lang}")

        # Translate text
        with st.spinner("Translating..."):
            system_msg = "You are a translator. Translate the following text to English." if detected_lang != 'en' else "Translate the following English text to Korean."
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": transcript_text}
                ]
            )
            translated_text = response.choices[0].message.content
            st.write("**Translated Text:**")
            st.write(translated_text)

        # Store in Neo4j
        with st.session_state.neo4j_driver.session() as session:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                CREATE (o:Original {text: $original_text})
                CREATE (t:Translation {text: $translated_text})
                MERGE (o)-[:DETECTED_AS]->(:Language {code: $language})
                MERGE (u)-[:REQUESTED]->(t)
                MERGE (t)-[:BASED_ON]->(o)
                """,
                user_id="anonymous",  # Replace with actual user ID if available
                original_text=transcript_text,
                translated_text=translated_text,
                language=detected_lang
            )

        # Download buttons
        st.download_button(label="Download Original Text", file_name="original_text.txt", data=transcript_text)
        st.download_button(label="Download Translated Text", file_name="translated_text.txt", data=translated_text)

# Visualize translation history
st.header("Translation History")
if st.button("Load History"):
    with st.session_state.neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (u:User)-[:REQUESTED]->(t:Translation)-[:BASED_ON]->(o:Original)
            RETURN o.text AS original, t.text AS translated
            """
        )
        history = result.values()
        for original, translated in history:
            st.write(f"**Original:** {original}")
            st.write(f"**Translated:** {translated}")