import os
import logging
from datetime import datetime
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for Streamlit
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Set Streamlit page configuration
try:
    st.set_page_config(
        page_title="IITD Buddy",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    logger.error(f"Failed to set page config: {e}")
    st.error("Failed to set page configuration. Please reload the app.")

# Caching resource initialization
@st.cache_resource
def init_qdrant():
    try:
        url = st.secrets["QDRANT_URL"]
        api_key = st.secrets["QDRANT_API_KEY"]
        return QdrantClient(url=url, api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        st.error("Failed to initialize Qdrant client. Check your configuration.")
        return None

@st.cache_resource
def init_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        st.error("Failed to initialize embedding model. Check your configuration.")
        return None

@st.cache_resource
def init_groq():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Groq: {e}")
        st.error("Failed to initialize Groq AI service. Check your configuration.")
        return None

# Initialize resources
qdrant_client = init_qdrant()
embedding_model = init_embedding_model()
groq_client = init_groq()

# Constants
COLLECTION_NAME = st.secrets.get("COLLECTION_NAME", "my_books")


# Custom CSS styling
st.markdown(
    """
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
        font-size: 1.1em;
    }
    .chat-message .timestamp {
        font-size: 0.8rem;
        color: #666;
    }
    .iitd-logo {
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 1000;
    }
    .structured-response {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        color: #1f77b4;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper functions
def query_llm(context: str, user_query: str, system_role: str) -> str:
    """Query Groq LLM for structured responses."""
    try:
        completion = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"""
                Based on the following information and query, provide a helpful response:

                Context:
                {context}

                User Query: {user_query}

                Please provide a structured response that addresses the query."""}
            ],
            temperature=1,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def search_qdrant(client: QdrantClient, collection_name: str, query: str, limit: int = 3):
    """Search resources in Qdrant."""
    try:
        query_vector = embedding_model.encode(query).tolist()
        hits = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        ).points

        if hits:
            context = "\n\n".join([
                f"Resource: {hit.payload['title']}\n"
                f"Description: {hit.payload.get('description', 'N/A')}"
                for hit in hits
            ])
            return context, hits
        return None, None
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None, None

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'counselling_messages' not in st.session_state:
    st.session_state.counselling_messages = []

# IITD logo
st.markdown(
    """
    <div class='iitd-logo'>
        <img src="https://upload.wikimedia.org/wikipedia/en/6/6d/Indian_Institute_of_Technology_Delhi_Logo.png" width="100">
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
with st.sidebar:
    st.markdown("### Navigation")
    rad = st.radio("", ["Home", "About Us", "COURSES OF STUDY", "BSW LINKS", "GUIDANCE AND COUNSELLING"])

# Navigation handling
if rad == "GUIDANCE AND COUNSELLING":
    st.title("🤝 Guidance and Counselling")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            ### 🎯 Mental Health and Wellness Resources
            Get information about counselling services, wellness programs, and support at IIT Delhi.
        """)

        with st.form("counselling_query_form"):
            user_query = st.text_input("What would you like to know about counselling services?")
            submitted = st.form_submit_button("Search 🔍")

            if submitted and user_query:
                context, hits = search_qdrant(
                    client=counselling_client,
                    collection_name=Config.COUNSELLING_COLLECTION,
                    query=user_query
                )
                if context and hits:
                    llm_response = query_llm(
                        context,
                        user_query,
                        """You are a compassionate counselling advisor at IIT Delhi. Provide supportive, 
                        understanding responses while maintaining professionalism."""
                    )
                    st.markdown("### 💭 Guidance Response")
                    st.markdown(f"<div class='structured-response'>{llm_response}</div>", unsafe_allow_html=True)

                    st.markdown("### 📚 Related Resources")
                    for hit in hits:
                        with st.expander(f"📘 {hit.payload['title']}"):
                            st.markdown(f"### Resource Information\n{hit.payload.get('description', 'N/A')}")

        st.markdown("""
            ### 🛑 Emergency Contacts
            * *IIT Delhi Counselling Service:* [Contact Number]
            * *24/7 Mental Health Helpline:* [Helpline Number]
            * *Student Wellness Centre:* [Contact Details]
        """)

    # with col2:
    #     st.markdown("### 💭 Wellness Assistant")
    #
    #     chat_container = st.container()
    #     with chat_container:
    #         for message in st.session_state.counselling_messages:
    #             st.markdown(f"<div class='chat-message {message['role']}'><div class='timestamp'>{message['timestamp']}</div>{message['content']}</div>", unsafe_allow_html=True)
    #
    #     with st.form("counselling_chat_form", clear_on_submit=True):
    #         chat_input = st.text_input("Ask about wellness resources:")
    #         if st.form_submit_button("Send"):
    #             if chat_input:
    #                 st.session_state.counselling_messages.append({
    #                     "role": "user",
    #                     "content": chat_input,
    #                     "timestamp": datetime.now().strftime("%H:%M")
    #                 })
    #
    #                 context, _ = search_qdrant(
    #                     client=counselling_client,
    #                     collection_name=Config.COUNSELLING_COLLECTION,
    #                     query=chat_input
    #                 )
    #                 ai_response = query_llm(
    #                     context or "",
    #                     chat_input,
    #                     """You are a compassionate counselling advisor at IIT Delhi."""
    #                 ) if context else "I couldn't find specific information. Please contact the wellness center."
    #
    #                 st.session_state.counselling_messages.append({
    #                     "role": "assistant",
    #                     "content": ai_response,
    #                     "timestamp": datetime.now().strftime("%H:%M")
    #                 })

                    st.experimental_rerun()

elif rad == "COURSES OF STUDY":
    st.title("📖 Course Search and Information")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🔍 Intelligent Course Search
        Ask detailed questions about courses, prerequisites, or get personalized recommendations!
        """)

        with st.form("query_form"):
            user_query = st.text_input("What would you like to know about courses?")
            submitted = st.form_submit_button("Search 🔍")

            if submitted and user_query:
                context, hits = search_qdrant(
                    client=course_client,
                    collection_name=Config.COURSE_COLLECTION,
                    query=user_query
                )
                if context and hits:
                    llm_response = query_llm(
                        context,
                        user_query,
                        """You are an academic advisor providing detailed information about courses."""
                    )
                    st.markdown("### 🤖 Detailed Course Analysis")
                    st.markdown(f"<div class='structured-response'>{llm_response}</div>", unsafe_allow_html=True)

                    st.markdown("### 📚 Related Course Details")
                    for hit in hits:
                        with st.expander(f"📘 {hit.payload['course_code']} - {hit.payload.get('title', 'N/A')}"):
                            st.markdown(f"""
                            ### Course Information
                            * *Course Code:* {hit.payload['course_code']}
                            * *Credits:* {json.dumps(hit.payload.get('credits', {}))}
                            * *Prerequisites:* {', '.join(hit.payload.get('prerequisites', []))}

                            ### Description
                            {hit.payload.get('description', 'N/A')}
                            """)

    # with col2:
    #     st.markdown("### 💭 AI Assistant")
    #
    #     chat_container = st.container()
    #     with chat_container:
    #         for message in st.session_state.messages:
    #             st.markdown(f"<div class='chat-message {message['role']}'><div class='timestamp'>{message['timestamp']}</div>{message['content']}</div>", unsafe_allow_html=True)
    #
    #     with st.form("chat_input_form", clear_on_submit=True):
    #         chat_input = st.text_input("Ask me anything:")
    #         if st.form_submit_button("Send"):
    #             if chat_input:
    #                 st.session_state.messages.append({
    #                     "role": "user",
    #                     "content": chat_input,
    #                     "timestamp": datetime.now().strftime("%H:%M")
    #                 })
    #
    #                 context, _ = search_qdrant(
    #                     client=course_client,
    #                     collection_name=Config.COURSE_COLLECTION,
    #                     query=chat_input
    #                 )
    #                 ai_response = query_llm(
    #                     context or "",
    #                     chat_input,
    #                     """You are an academic advisor at IIT Delhi."""
    #                 ) if context else "I couldn't find specific course information. Please try rephrasing your query."
    #
    #                 st.session_state.messages.append({
    #                     "role": "assistant",
    #                     "content": ai_response,
    #                     "timestamp": datetime.now().strftime("%H:%M")
    #                 })

                    st.experimental_rerun()

elif rad == "Home":
    st.title("🏠 Welcome to IITD Buddy")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
        <h3>Your One-Stop Solution for IIT Delhi Journey</h3>
        <p>Get instant access to resources, guidance, and support!</p>
    </div>
    """, unsafe_allow_html=True)

    # Create three columns for features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📚 Academic Resources")
        st.info("Access course materials, lecture notes, and past papers")

    with col2:
        st.markdown("### 🎯 Career Guidance")
        st.info("Connect with mentors and explore opportunities")

    with col3:
        st.markdown("### 💡 Student Support")
        st.info("Get help with academic and personal challenges")
else:
    st.title(f"📁 {rad}")
    st.markdown("Content coming soon...")
