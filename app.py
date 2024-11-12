import os
import tiktoken
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
import openai
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()


# Set OpenAI API key as an environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Transcribe the audio file to text using OpenAI API
def transcribe_audio(uploaded_audio_file, model="whisper-1"):
  # Save the uploaded file temporarily
  with open("temp_audio.mp3", "wb") as f:
    f.write(uploaded_audio_file.getvalue())
  
  # Now using the saved file path
  with open("temp_audio.mp3", "rb") as audio:
    response = openai.audio.transcriptions.create(file=audio, model=model)
    transcript_text = response.text
    output_file = "files/transcripts/transcript.txt"
  # Create the directory for the output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Write the transcript to the output file
    with open(output_file, "w") as file:
      file.write(transcript_text)
  return transcript_text

#Create the Document Search
transcript_text = "files/transcripts/transcript.txt"
def create_qa_model(transcript_text):
  loader = TextLoader(transcript_text)
  docs = loader.load()
  # Create a new DocArrayInMemorySearch instance from the specified documents and embeddings
  db = DocArrayInMemorySearch.from_documents(docs, OpenAIEmbeddings())
  # Convert the DocArrayInMemorySearch instance to a retriever. Assign to retriever
  retriever = db.as_retriever()
  llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.0)
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True) 

  return qa

def main():
  st.title("Video-to-Text Q&A Bot")

  audio_file = st.file_uploader("Upload your video here (e.g., .mp3)", type=["mp4"])

  if audio_file:
    with st.spinner('Transcribing audio...'):
      transcript = transcribe_audio(audio_file)
      st.success("Transcription complete!")
      st.text_area("Transcript:", transcript, height=300)
      qa_model = create_qa_model(transcript_text) 

      question = st.text_input("Ask a question about the transcription:")
      
      if st.button("Get Answer"):
        if question:
          answer = qa_model.invoke(question)
          #answer = qa_model({"question": question})
          st.write("Answer:", answer['result'])
  else:
    st.info("Please upload an audio file for transcription.")

if __name__ == "__main__":
  main()