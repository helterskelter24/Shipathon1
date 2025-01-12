import os
import warnings
import streamlit as st
import pandas as pd
from datetime import datetime
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

# Disable file watcher to avoid inotify watch limit issue
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set page configuration
st.set_page_config(
    page_title="IITD Buddy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize clients with proper error handling
@st.cache_resource
def init_qdrant():
    try:
        return QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to initialize Qdrant: {str(e)}")
        return None

@st.cache_resource
def init_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to initialize embedding model: {str(e)}")
        return None

@st.cache_resource
def init_groq():
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception as e:
        st.error(f"Failed to initialize Groq: {str(e)}")
        return None

# Initialize services
client = init_qdrant()
embedding_model = init_embedding_model()
groq_client = init_groq()

# Custom CSS remains the same
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    .chat-message .timestamp {
        font-size: 0.8rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

def query_groq_llm(context: str, user_query: str) -> str:
    """Query Groq LLM with context and user query"""
    if not groq_client:
        return "I apologize, but the AI service is currently unavailable."
    
    try:
        completion = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": """You are IITD Buddy, an AI assistant for IIT Delhi students. 
                You help students with course-related queries and provide information about academics at IIT Delhi. 
                Be concise, friendly, and accurate in your responses."""},
                {"role": "user", "content": f"""
                Based on the following course information and user query, provide a helpful response:
                
                Search Results:
                {context}
                
                User Query: {user_query}
                
                Provide a clear, conversational response addressing the query."""}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def search_courses(query: str):
    """Search courses in Qdrant and return results"""
    if not all([client, embedding_model]):
        st.error("Search services are currently unavailable.")
        return None, None
        
    try:
        query_vector = embedding_model.encode(query).tolist()
        hits = client.query_points(
            collection_name=st.secrets["COLLECTION_NAME"],
            query=query_vector,
            limit=3
        ).points
        
        if hits:
            context = "\n\n".join([
                f"Course: {hit.payload['course_code']} - {hit.payload.get('title', 'N/A')}\n"
                f"Credits: {json.dumps(hit.payload.get('credits', {}))}\n"
                f"Prerequisites: {', '.join(hit.payload.get('prerequisites', []))}\n"
                f"Description: {hit.payload.get('description', 'N/A')}"
                for hit in hits
            ])
            return context, hits
        return None, None
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None, None

# Rest of your code remains the same...

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/6/6d/Indian_Institute_of_Technology_Delhi_Logo.png", width=150)
    st.markdown("### Navigation")
    rad = st.radio("", [
        "Home",
        "About Us",
        "COURSES OF STUDY",
        "BSW LINKS",
        "GUIDANCE AND COUNSELLING"
    ])

# Main content area
if rad == "COURSES OF STUDY":
    st.title("📖 Course Search and Information")
    
    # Course search interface
    st.markdown("""
    ### 🔍 Intelligent Course Search
    Enter your query about courses - ask about prerequisites, content, or get recommendations!
    """)
    
    with st.form("query_form"):
        user_query = st.text_input("What would you like to know about courses?")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button("Search 🔍", use_container_width=True)
        
        if submitted and user_query:
            context, hits = search_courses(user_query)
            if context and hits:
                llm_response = query_groq_llm(context, user_query)
                st.markdown("### 🤖 AI Response")
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
                    {llm_response}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📚 Related Courses")
                for hit in hits:
                    with st.expander(f"📘 {hit.payload['course_code']} - {hit.payload.get('title', 'N/A')}"):
                        st.markdown(f"""
                        * *Credits:* {json.dumps(hit.payload.get('credits', {}))}
                        * *Prerequisites:* {', '.join(hit.payload.get('prerequisites', []))}
                        
                        *Description:*
                        {hit.payload.get('description', 'N/A')}
                        """)

elif rad == "Home":
    st.title("🏠 Welcome to IITD Buddy")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
        <h3>Your AI-Powered Academic Assistant</h3>
        <p>Get instant answers about courses, academic policies, and more!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📚 Course Information")
        st.info("Get detailed course information and prerequisites")
    with col2:
        st.markdown("### 🤖 AI Assistant")
        st.info("Chat with our AI for instant academic guidance")
    with col3:
        st.markdown("### 📋 Resources")
        st.info("Access important links and documents")

else:
    # Other sections remain the same as before
    st.title(f"📑 {rad}")
    st.markdown("Content coming soon...")

# AI Chatbot (present on all pages)
st.markdown("---")
st.markdown("### 🤖 Chat with IITD Buddy")

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message['role']}">
            <div class="timestamp">{message['timestamp']}</div>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)

# Chat input
with st.form("chat_input_form", clear_on_submit=True):
    chat_input = st.text_input("Ask me anything about courses and academics:", key="chat_input")
    if st.form_submit_button("Send", use_container_width=True):
        if chat_input:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": chat_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # Search courses and get AI response
            context, _ = search_courses(chat_input)
            if context:
                ai_response = query_groq_llm(context, chat_input)
            else:
                ai_response = "I apologize, but I couldn't find specific course information for your query. Could you please rephrase or ask something else?"
            
            # Add AI response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            st.rerun()
