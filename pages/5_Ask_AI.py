import streamlit as st
import google.generativeai as genai


def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "parts": ["Hello, how can I help you?"]}
        ]
    if "generating" not in st.session_state:
        st.session_state["generating"] = (
            False  # Add a boolean key to ensure that response can only be generated if not already generating.
        )


def display_chat_history():
    for message in st.session_state["chat_history"]:
        if message["role"] == "assistant":
            st.chat_message("ai").write(message["parts"][0])

        if message["role"] == "user":
            st.chat_message("human").write(message["parts"][0])


def run():
    st.set_page_config(page_title="Ask AI")
    st.header("Ask :blue[AI]")

    initialize_session_state()
    display_chat_history()  # This will redisplay the chat history.

    prompt = st.chat_input("Add your prompt...")

    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("Please enter your Gemini API key in Streamlit secrets.")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    if (
        prompt and not st.session_state["generating"]
    ):  # If the user put in a prompt, and we're not already generating, then generate a response.
        st.session_state["generating"] = (
            True  # Set this to true to prevent duplicate response generation
        )
        st.chat_message("human").write(prompt)
        st.session_state["chat_history"].append({"role": "user", "parts": [prompt]})

        with st.chat_message("ai"):
            with st.spinner("Generating response..."):
                try:
                    response = model.generate_content(st.session_state["chat_history"])
                    st.write(response.text)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "parts": [response.text]}
                    )

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error(
                        "Check that your API Key is valid, the model is available, and the message format is correct."
                    )
        st.session_state["generating"] = False  # After the response, set this to false.


run()
