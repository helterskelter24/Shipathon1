import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

# Page configuration
st.set_page_config(
    page_title="Course Search - IITD",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS remains the same
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Arial', sans-serif;
    }
    .main {
        padding: 2rem;
        color: #000000;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        color: #000000;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-bottom: 1rem;
        color: #000000;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

def check_secrets():
    """Verify all required secrets are present."""
    required_secrets = {
        "QDRANT_URL": "Qdrant database URL",
        "QDRANT_API_KEY": "Qdrant API key",
        "GROQ_API_KEY": "Groq API key",
        "COLLECTION_NAME": "Qdrant collection name"
    }
    
    missing_secrets = []
    for secret, description in required_secrets.items():
        if secret not in st.secrets:
            missing_secrets.append(f"{secret} ({description})")
    
    return missing_secrets

# Initialize connection to Qdrant with proper error handling
@st.cache_resource
def init_qdrant():
    try:
        return QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"]
        )
    except KeyError as e:
        st.error(f"Missing required secret: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Failed to initialize Qdrant: {str(e)}")
        return None

# Initialize Sentence Transformer
@st.cache_resource
def init_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to initialize embedding model: {str(e)}")
        return None

# Initialize Groq client with proper error handling
@st.cache_resource
def init_groq():
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except KeyError as e:
        st.error(f"Missing required secret: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Failed to initialize Groq: {str(e)}")
        return None

def query_groq_llm(context: str, user_query: str, groq_client) -> str:
    try:
        completion = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system",
                 "content": "You are a helpful academic advisor. Provide clear, concise information about courses based on the search results."},
                {"role": "user", "content": f"""
                Based on the following course information and user query, provide a helpful response:

                Search Results:
                {context}

                User Query: {user_query}

                Please provide a clear, structured response addressing the query."""}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error querying Groq LLM: {str(e)}"

def display_course_card(hit, index):
    with st.container():
        st.markdown(f"""
        <div style="
            padding: 1rem;
            border-radius: 5px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            color: #000000;
        ">
            <h4 style="color: #000000; margin: 0;">{hit.payload['course_code']} - {hit.payload.get('title', 'N/A')}</h4>
            <p style="color: #000000; font-size: 0.9rem;">Score: {hit.score:.3f}</p>
            <p style="color: #000000;"><strong>Credits:</strong> {json.dumps(hit.payload.get('credits', {}))}</p>
            <p style="color: #000000;"><strong>Prerequisites:</strong> {', '.join(hit.payload.get('prerequisites', []))}</p>
            <p style="color: #000000;"><strong>Description:</strong> {hit.payload.get('description', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("🔍 Enhanced Course Search")
    
    # Check for missing secrets first
    missing_secrets = check_secrets()
    if missing_secrets:
        st.error("Missing required secrets:")
        for secret in missing_secrets:
            st.write(f"❌ {secret}")
        st.info("Please add these secrets in your Streamlit Cloud dashboard or .streamlit/secrets.toml file.")
        return
    
    # Initialize services
    with st.spinner("Initializing services..."):
        qdrant_client = init_qdrant()
        embedding_model = init_embedding_model()
        groq_client = init_groq()

        if not all([qdrant_client, embedding_model, groq_client]):
            st.error("Failed to initialize one or more services. Please check the logs above.")
            return

    st.markdown("""
    <p style="color: #000000;">
    Welcome to the Course Search platform! Enter your query about courses below to get detailed information 
    and personalized recommendations from our AI advisor.
    </p>
    """, unsafe_allow_html=True)

    # Search interface
    with st.form("query_form", clear_on_submit=False):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_query = st.text_input("Enter your query about courses:",
                                     placeholder="e.g., What are the prerequisites for Machine Learning courses?")
        with col2:
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)

        if submitted and user_query:
            try:
                # Convert query to vector
                query_vector = embedding_model.encode(user_query).tolist()

                # Search Qdrant using the collection name from secrets
                hits = qdrant_client.query_points(
                    collection_name=st.secrets["COLLECTION_NAME"],
                    query=query_vector,
                    limit=3
                ).points

                if hits:
                    # Format context from search results
                    context = "\n\n".join([
                        f"Course: {hit.payload['course_code']} - {hit.payload.get('title', 'N/A')}\n"
                        f"Credits: {json.dumps(hit.payload.get('credits', {}))}\n"
                        f"Prerequisites: {', '.join(hit.payload.get('prerequisites', []))}\n"
                        f"Description: {hit.payload.get('description', 'N/A')}"
                        for hit in hits
                    ])

                    # Get LLM response
                    with st.spinner("🤔 Analyzing your query..."):
                        llm_response = query_groq_llm(context, user_query, groq_client)
                        st.markdown("### 🤖 AI Advisor Response")
                        st.markdown(f"""
                        <div style="
                            padding: 1.5rem;
                            border-radius: 5px;
                            background-color: #f8f9fa;
                            border-left: 5px solid #FF4B4B;
                            margin: 1rem 0;
                            color: #000000;
                        ">
                            {llm_response}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('<h3 style="color: #000000;">📚 Detailed Search Results</h3>', unsafe_allow_html=True)
                    for i, hit in enumerate(hits, 1):
                        display_course_card(hit, i)
                else:
                    st.warning("No results found for your query. Try rephrasing or using different keywords.")

            except Exception as e:
                st.error(f"Error during search: {str(e)}")

if __name__ == "__main__":
    main()
