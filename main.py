import streamlit as st
import pandas as pd
from datetime import datetime
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

# Set page configuration
st.set_page_config(
    page_title="IITD Buddy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Configuration class with error handling
try:
    class Config:
        @staticmethod
        def get_secrets():
            required_secrets = [
                "COURSE_QDRANT_URL",
                "COURSE_QDRANT_KEY",
                "COURSE_COLLECTION",
                "COUNSELLING_QDRANT_URL",
                "COUNSELLING_QDRANT_KEY",
                "COUNSELLING_COLLECTION",
                "GROQ_API_KEY"
            ]
            
            missing_secrets = []
            secrets_dict = {}
            
            for secret in required_secrets:
                value = st.secrets.get(secret)
                if not value:
                    missing_secrets.append(secret)
                secrets_dict[secret] = value
            
            if missing_secrets:
                st.error(f"Missing required secrets: {', '.join(missing_secrets)}")
                st.stop()
            
            return secrets_dict

        secrets = get_secrets()
        COURSE_QDRANT_URL = secrets["COURSE_QDRANT_URL"]
        COURSE_QDRANT_KEY = secrets["COURSE_QDRANT_KEY"]
        COURSE_COLLECTION = secrets["COURSE_COLLECTION"]
        COUNSELLING_QDRANT_URL = secrets["COUNSELLING_QDRANT_URL"]
        COUNSELLING_QDRANT_KEY = secrets["COUNSELLING_QDRANT_KEY"]
        COUNSELLING_COLLECTION = secrets["COUNSELLING_COLLECTION"]
        GROQ_API_KEY = secrets["GROQ_API_KEY"]

except Exception as e:
    st.error(f"Error initializing configuration: {str(e)}")
    st.stop()

# Initialize clients with error handling
try:
    course_client = QdrantClient(url=Config.COURSE_QDRANT_URL, api_key=Config.COURSE_QDRANT_KEY)
    counselling_client = QdrantClient(url=Config.COUNSELLING_QDRANT_URL, api_key=Config.COUNSELLING_QDRANT_KEY)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=Config.GROQ_API_KEY)
except Exception as e:
    st.error(f"Error initializing clients: {str(e)}")
    st.stop()

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

# Helper functions with error handling
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
    """Search resources in Qdrant with error handling."""
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

# Rest of your code remains the same...
# [Include all the navigation handling code (Home, About Us, etc.) here exactly as it was before]
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
            * IIT Delhi Counselling Service: [Contact Number]
            * 24/7 Mental Health Helpline: [Helpline Number]
            * Student Wellness Centre: [Contact Details]
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

                    # st.experimental_rerun()


# elif rad == "ABOUT":
#     st.title("🎓 About Our Learning Platform")
    
#     # Welcome Section
#     st.markdown("""
#     ### 👋 Welcome
#     Welcome to our platform, where we are committed to providing accessible and insightful resources 
#     for students and professionals alike. Our mission is to create an environment where learning 
#     thrives and knowledge is accessible to all.
#     """)
    
#     # Course Offerings
#     st.markdown("### 📚 Our Course Offerings")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         #### MCL Courses
#         Our primary curriculum is built around MCL courses, offering comprehensive learning paths 
#         for those eager to enhance their knowledge in various fields.
#         """)
    
#     with col2:
#         st.markdown("""
#         #### BSW Program (APL100 Series)
#         We offer a unique collection of Bachelor of Social Work courses through our APL100 series, 
#         designed to equip students with:
#         * Practical skills for real-world application
#         * Theoretical knowledge for deep understanding
#         * Professional competencies for success
#         """)
    
#     # Current Status
#     st.markdown("### 🚀 Platform Development")
    
#     with st.container():
#         st.info("""
#         **Current Status Update**
        
#         Our platform is in active development. In just 48 hours, we've successfully:
#         * Developed a working model
#         * Established a foundation database
#         * Created an intuitive user interface
        
#         While our current database is limited, we're committed to continuous improvement and expansion.
#         """)
    
#     # Future Plans
#     st.markdown("### 🔮 Future Developments")
    
#     with st.container():
#         st.success("""
#         We are actively working on:
#         * Expanding our course database
#         * Enhancing platform functionality
#         * Improving user experience
#         * Adding more comprehensive resources
        
#         Your support and understanding during this growth phase are greatly appreciated.
#         """)
    
#     # Footer Note
#     st.markdown("---")
#     st.markdown("""
#     💡 *We value your feedback and suggestions as we continue to grow and improve our platform. 
#     Thank you for being part of our learning community!*
#     """)


elif rad == "About Us":
    st.title("🎓 About Our Learning Platform")
    
    # Welcome Section
    st.markdown("""
    ### 👋 Welcome
    Welcome to our platform, where we are committed to providing accessible and insightful resources 
    for students and professionals alike. Our mission is to create an environment where learning 
    thrives and knowledge is accessible to all.
    """)
    
    # Course Offerings
    st.markdown("### 📚 Our Course Offerings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### MCL Courses
        Our primary curriculum is built around MCL courses, offering comprehensive learning paths 
        for those eager to enhance their knowledge in various fields.
        """)
    
    with col2:
        st.markdown("""
        #### BSW Program (APL100 Series)
        We offer a unique collection of Bachelor of Social Work courses through our APL100 series, 
        designed to equip students with:
        * Practical skills for real-world application
        * Theoretical knowledge for deep understanding
        * Professional competencies for success
        """)
    
    # Current Status
    st.markdown("### 🚀 Platform Development")
    
    with st.container():
        st.info("""
        **Current Status Update**
        
        Our platform is in active development. In just 48 hours, we've successfully:
        * Developed a working model
        * Established a foundation database
        * Created an intuitive user interface
        
        While our current database is limited, we're committed to continuous improvement and expansion.
        """)
    
    # Future Plans
    st.markdown("### 🔮 Future Developments")
    
    with st.container():
        st.success("""
        We are actively working on:
        * Expanding our course database
        * Enhancing platform functionality
        * Improving user experience
        * Adding more comprehensive resources
        
        Your support and understanding during this growth phase are greatly appreciated.
        """)
    
    # Footer Note
    st.markdown("---")
    st.markdown("""
    💡 *We value your feedback and suggestions as we continue to grow and improve our platform. 
    Thank you for being part of our learning community!*
    """)
    
    # Team Section
    st.markdown("### 👥 Meet Our Team")
    
   # [Previous code remains the same until the team section]

    # Team Section
    # st.markdown("### 👥 Meet Our Team")
    
    # Adding custom CSS for team section
    st.markdown("""
    <style>
    .team-grid {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 20px;
        padding: 20px 0;
    }
    .team-member {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        width: 200px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .team-member:hover {
        transform: translateY(-5px);
        transition: transform 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    # Team member details with corrected spelling
    st.markdown("""
    <div class="team-grid">
        <div class="team-member">
            <h3>👤 Lavanya Singla</h3>
            <p>Role: Developer</p>
        </div>
        <div class="team-member">
            <h3>👤 Aaditya Mehar</h3>
            <p>Role: Developer</p>
        </div>
        <div class="team-member">
            <h3>👤 Manya Gangwar</h3>
            <p>Role: Developer</p>
        </div>
        <div class="team-member">
            <h3>👤 Saptarshi Banerjee </h3>
            <p>Role: Developer</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# [Rest of the code remains the same]

    # Copyright
    st.markdown("---")
    st.markdown("© 2024 Learning Platform Team. All rights reserved.")

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
                            * Course Code: {hit.payload['course_code']}
                            * Credits: {json.dumps(hit.payload.get('credits', {}))}
                            * Prerequisites: {', '.join(hit.payload.get('prerequisites', []))}

                            ### Description
                            {hit.payload.get('description', 'N/A')}
                            """)

# elif rad == "COURSES OF STUDY":
#     st.title("📖 Course Search and Information")

#     col1, col2 = st.columns([2, 1])

#     with col1:
#         st.markdown("""
#         ### 🔍 Intelligent Course Search
#         Ask detailed questions about courses, prerequisites, or get personalized recommendations!
#         """)

#         with st.form("query_form"):
#             user_query = st.text_input("What would you like to know about courses?")
#             submitted = st.form_submit_button("Search 🔍")

#             if submitted and user_query:
#                 context, hits = search_qdrant(
#                     client=course_client,
#                     collection_name=Config.COURSE_COLLECTION,
#                     query=user_query
#                 )
#                 if context and hits:
#                     llm_response = query_llm(
#                         context,
#                         user_query,
#                         """You are an academic advisor providing detailed information about courses."""
#                     )
#                     st.markdown("### 🤖 Detailed Course Analysis")
#                     st.markdown(f"<div class='structured-response'>{llm_response}</div>", unsafe_allow_html=True)

#                     st.markdown("### 📚 Related Course Details")
#                     for hit in hits:
#                         with st.expander(f"📘 {hit.payload['course_code']} - {hit.payload.get('title', 'N/A')}"):
#                             st.markdown(f"""
#                             ### Course Information
#                             * Course Code: {hit.payload['course_code']}
#                             * Credits: {json.dumps(hit.payload.get('credits', {}))}
#                             * Prerequisites: {', '.join(hit.payload.get('prerequisites', []))}

#                             ### Description
#                             {hit.payload.get('description', 'N/A')}
#                             """)

#     # with col2:
#     #     st.markdown("### 💭 AI Assistant")
#     #
#     #     chat_container = st.container()
#     #     with chat_container:
#     #         for message in st.session_state.messages:
#     #             st.markdown(f"<div class='chat-message {message['role']}'><div class='timestamp'>{message['timestamp']}</div>{message['content']}</div>", unsafe_allow_html=True)
#     #
#     #     with st.form("chat_input_form", clear_on_submit=True):
#     #         chat_input = st.text_input("Ask me anything:")
#     #         if st.form_submit_button("Send"):
#     #             if chat_input:
#     #                 st.session_state.messages.append({
#     #                     "role": "user",
#     #                     "content": chat_input,
#     #                     "timestamp": datetime.now().strftime("%H:%M")
#     #                 })
#     #
#     #                 context, _ = search_qdrant(
#     #                     client=course_client,
#     #                     collection_name=Config.COURSE_COLLECTION,
#     #                     query=chat_input
#     #                 )
#     #                 ai_response = query_llm(
#     #                     context or "",
#     #                     chat_input,
#     #                     """You are an academic advisor at IIT Delhi."""
#     #                 ) if context else "I couldn't find specific course information. Please try rephrasing your query."
#     #
#     #                 st.session_state.messages.append({
#     #                     "role": "assistant",
#     #                     "content": ai_response,
#     #                     "timestamp": datetime.now().strftime("%H:%M")
#     #                 })

#                     st.rerun()


elif rad == "BSW LINKS":
    st.title("🔍 GET TO KNOW ABOUT BSW WEBSITE")
    # Configuration for BSW LINKS
    qdrant_url = st.secrets["COURSE_QDRANT_URL"]
    qdrant_api_key = st.secrets["COURSE_QDRANT_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
    collection_name = "APL_LINKS_APL"

    # Initialize clients
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=groq_api_key)

    def format_links(source_data):
        """
        Format source links properly based on the data structure in Qdrant
        Returns a list of properly formatted links
        """
        if isinstance(source_data, list):
            return source_data
        elif isinstance(source_data, str):
            try:
                # Try to parse if it's a JSON string
                return json.loads(source_data)
            except json.JSONDecodeError:
                # If it's a single string, return it as a single-item list
                return [source_data]
        return []

    def query_groq_llm(context: str, user_query: str) -> str:
        """Query Groq LLM with context and user query"""
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
                    """}
                ],
                temperature=1,
                max_tokens=500
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error querying Groq LLM: {str(e)}"

    # User input for search query
    user_query = st.text_input("Enter your query related to BSW resources:")
    if user_query:
        # Qdrant search
        query_vector = embedding_model.encode(user_query).tolist()
        try:
            hits = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=5
            ).points

            if hits:
                context = "\n\n".join([
                    f"Resource Title: {hit.payload.get('name', 'N/A')}\n"
                    f"Year: {hit.payload.get('Year', 'N/A')}"
                    f"Year: {hit.payload.get('Year', 'N/A')}\n"
                    f"Links: {', '.join(format_links(hit.payload.get('source', [])))}"
                    for hit in hits
                ])
                response = query_groq_llm(context, user_query)

                st.markdown("### 🤖 LLM Response")
                st.markdown(f"<div class='structured-response'>{response}</div>", unsafe_allow_html=True)

                st.markdown("### 📚 Related Resources")
                for hit in hits:
                    with st.expander(f"📘 {hit.payload.get('title', 'N/A')}"):
                        st.markdown(f"""
                        **Description:** {hit.payload.get('description', 'N/A')}
                        **Links:** {', '.join(format_links(hit.payload.get('source', [])))}
                        **Sem:**{",".join(format_links(hit.payload.get('SEM',[])))}
                        **Year:**{",".join(format_links(hit.payload.get('Year',[])))}
                        """)

            else:
                st.warning("No relevant resources found.")
        except Exception as e:
            st.error(f"Error querying Qdrant: {str(e)}")

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
