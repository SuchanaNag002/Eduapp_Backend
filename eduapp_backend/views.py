from io import BytesIO
import os
import json
from .models import Video
import sqlite3
from venv import logger
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from .youtube_transcript import extract_transcript  
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#quiz functions

def generate_text(topic):
    if not os.getenv("GOOGLE_API_KEY"):
        return "Missing Google API Key! Set the GOOGLE_API_KEY environment variable."
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
        prompt = '''Please format the response as a Stringified JSON array of objects, where each object has the following properties:
- `QuestionNumber`: An integer indicating the question's numerical order (starting from 1). Use the format `question_number=n` in the prompt to specify this explicitly.
- `Question`: The actual question text, limited to 50 characters.
- `A`, `B`, `C`, and `D`: Four answer options, each also limited to 50 characters.
- `CorrectAnswer`: The single correct answer (A, B, C, or D).
- `Explanation`: A clear and concise explanation of the correct answer, adhering to a 200-character limit.
Do not use Markdown styles. No added text or replies. Should be parsable by JSON.parse() . No styling. no ```* .
**Example:**
[{"QuestionNumber": 1,"Question": "Capital of France?","A": "London","B": "Berlin","C": "Paris","D": "Rome","CorrectAnswer": "C","Explanation": "Paris is the capital of France."
  }, // ... more questions ]'''
        prompt += f'''\nPlease generate a questionnaire on the topic: **{topic}**\n**Number of questions:** 10'''
        generated_text = llm.invoke(prompt).content.strip()
        return json.loads(generated_text)
    except Exception as e:
        return f"An error occurred: {str(e)}"


@csrf_exempt
def generate_questionnaire(request):
    if request.method == "GET":
        return render(request, 'eduapp_backend/generate_questionnaire.html')
    elif request.method == "POST":
        try:
            data = json.loads(request.body)
            paragraph = data.get("paragraph")
            if not paragraph:
                return JsonResponse({"error": "Missing paragraph in request body"}, status=400)
            generated_questions = generate_text(paragraph)
            return JsonResponse(generated_questions, safe=False)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data in request body"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)
    
#ask pdf functions

def get_conversational_chain():
    prompt_template = """
        Answer the question in detail as much as possible from the provided context, make sure to provide all the 
        details, if the answer is not in the provided context just say, "answer is not available in the context", do not
        provide the wrong answer\n\n
        Context:\n{context}?\n
        Question:\n{question}\n

        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_answer(user_question, text_chunks):
    index_name = "pdf-chatbot"
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    # Embed the user's question
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    query_embedding = embeddings_model.embed_query(user_question)

    # Query the Pinecone index for similar vectors
    response = index.query(
        vector=[query_embedding],
        top_k=5,
        include_values=True
    )

    # If matches are found, use conversational model to generate response
    if response.get('matches'):
        chain = get_conversational_chain()
        docs = [Document(page_content=chunk) for chunk in text_chunks]
        response_from_chain = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        print(response_from_chain["output_text"])
        return response_from_chain["output_text"]
    else:
        return "Sorry, I couldn't find relevant information to answer your question."


def extract_text_chunks(pdf_file):
    text = ""
    if isinstance(pdf_file, bytes):
        pdf_file = BytesIO(pdf_file)

    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)

@csrf_exempt
def ask_question(request):
    print(request)
    if request.method == 'POST':
        # Retrieve PDF file and user question from the request
        pdf_file = request.FILES.get('pdf_file')
        user_question = request.POST.get('question')
        
        print(f"Received PDF file: {pdf_file}")
        print(f"Received user question: {user_question}")

        # Validate input
        if not pdf_file:
            error_message = 'Please upload a PDF file.'
            logger.error(error_message)
            return JsonResponse({'error': error_message}, status=400)
        if not user_question:
            error_message = 'Please enter a question.'
            logger.error(error_message)
            return JsonResponse({'error': error_message}, status=400)
        
        # Process PDF file and user question
        try:
            text_chunks = extract_text_chunks(pdf_file)
            logger.info("Successfully extracted text chunks from PDF")
            response = generate_answer(user_question, text_chunks)
            logger.info("Successfully generated answer")
            return JsonResponse({'response': response})
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            logger.error(error_message)
            return JsonResponse({'error': error_message}, status=500)
    else:
        return render(request, 'eduapp_backend/ask_pdf.html')


#video transcription functions
def fetch_transcript(video_id):
    conn = sqlite3.connect('youtube_transcripts.db')
    c = conn.cursor()
    c.execute("SELECT transcript FROM transcripts WHERE video_id = ?", (video_id,))
    transcript_text = c.fetchone()[0]
    conn.close()
    return transcript_text

def generate_notes(transcript_text, subject):
    prompt = """
        Title: Detailed Notes on {subject} from YouTube Video Transcript

        As an expert in {subject}, your task is to provide detailed notes based on the transcript of a YouTube video I'll provide. Assume the role of a student and generate comprehensive notes covering the key concepts discussed in the video.

        Your notes should:

        - Analyze and explain the main ideas, theories, or concepts presented in the video.
        - Provide examples, illustrations, or case studies to support the understanding of the topic.
        - Offer insights or practical applications of the subject matter discussed.
        - Use clear language and concise explanations to facilitate learning.

        Please provide the YouTube video transcript, and I'll generate the detailed notes on {subject} accordingly.
    """.format(subject=subject)

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt + transcript_text)
    return response.text


from .models import Video

def convert(request):
    if request.method == 'POST':
        youtube_link = request.POST.get('youtube_link')
        subject = request.POST.get('subject')

        video_id = youtube_link.split("=")[-1]
        thumbnail_url = f"http://img.youtube.com/vi/{video_id}/0.jpg"

        transcript_text = extract_transcript(youtube_link)
        if transcript_text:
            notes = generate_notes(transcript_text, subject)
            video = Video.objects.create(
                url=youtube_link,
                thumbnail_url=thumbnail_url,
                notes=notes
            )
            return render(request, 'eduapp_backend/youtube_notes.html', {'notes': notes})
        else:
            return JsonResponse({"error": "Failed to get transcript_text"}, status=400)
    else:
        return render(request, 'eduapp_backend/youtube_notes.html')
