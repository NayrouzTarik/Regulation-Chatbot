import streamlit as st
from main_script import main_pipeline, initialize_retrieval_system, pdf_path, db_path, embedding
import re

# Use st.cache_resource to ensure the retrieval system is only initialized once
@st.cache_resource
def initialize_system():
    return initialize_retrieval_system(pdf_path, db_path, embedding)

def strip_formatting(text):
    # Remove HTML tags and Markdown formatting
    clean_text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    clean_text = re.sub(r'\[.*?\]', '', clean_text)  # Remove Markdown links
    clean_text = re.sub(r'(\*|_|~|`|>)', '', clean_text)  # Remove Markdown formatting characters
    return clean_text

def main():
    st.set_page_config(page_title="EU AI Act Guardian", page_icon="🤖")

    st.markdown(
        """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
        }
        .user-bubble, .bot-bubble {
            border-radius: 15px;
            padding: 10px 15px;
            margin: 10px 0;
            word-wrap: break-word;
        }
        .user-bubble {
            background-color: #d1e7dd;
            text-align: right;
            color: #0f5132;
        }
        .bot-bubble {
            background-color: #f8d7da;
            text-align: left;
            color: #842029;
        }
        .context-bubble {
            background-color: #f1f1f1;
            text-align: left;
            color: #333;
            border: 1px solid #ccc;
        }
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            padding: 10px 0;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.title("EU AI Act Guardian")
    st.write("Your EU AI Act Guardian is here to help you understand more about the legislation and make it clear for you.")

    # Initialize the retrieval system
    vector_store = initialize_system()

    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
        st.session_state.current_session = None

    st.sidebar.markdown('<div class="sidebar-title">Sessions</div>', unsafe_allow_html=True)
    session_names = list(st.session_state.sessions.keys())

    if st.sidebar.button("New Chat"):
        new_session_name = f"Session {len(st.session_state.sessions) + 1}"
        st.session_state.sessions[new_session_name] = []
        st.session_state.current_session = new_session_name

    session_choice = st.sidebar.selectbox("Select a session:", session_names, key="session_select")

    if session_choice:
        st.session_state.current_session = session_choice

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if st.session_state.current_session:
        for chat in st.session_state.sessions[st.session_state.current_session]:
            if chat["role"] == "user":
                st.markdown(f'<div class="user-bubble"><strong>You:</strong> {chat["text"]}</div>', unsafe_allow_html=True)
            elif chat["role"] == "bot":
                st.markdown(f'<div class="bot-bubble"><strong>AI Act Guardian:</strong> {strip_formatting(chat["text"])}</div>', unsafe_allow_html=True)
            elif chat["role"] == "context":
                st.markdown(f'<div class="context-bubble"><strong>Context:</strong> {strip_formatting(chat["text"])}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    with st.form(key="query_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="user_input", height=100)  
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input.strip():
        # Trigger main_pipeline function with the initialized retrieval_chain
        result, context = main_pipeline(user_input)

        st.session_state.sessions[st.session_state.current_session].append({"role": "user", "text": user_input.strip()})
        st.session_state.sessions[st.session_state.current_session].append({"role": "bot", "text": result})
        st.session_state.sessions[st.session_state.current_session].append({"role": "context", "text": context})

        st.markdown(f'<div class="user-bubble"><strong>You:</strong> {user_input.strip()}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-bubble"><strong>AI Act Guardian:</strong> {strip_formatting(result)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="context-bubble"><strong>Context:</strong> {strip_formatting(context)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
