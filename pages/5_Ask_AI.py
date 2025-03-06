import streamlit as st
import google.generativeai as genai
from PIL import Image
import base64
from io import BytesIO

# Configure the app
st.set_page_config(
    page_title="Ask AI",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Setup API key from secrets
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# Initialize the Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.3,
    },
)


# Helper functions
def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))


def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode("utf-8")


# Function to convert messages for Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None

    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append({"text": content["text"]})
            elif content["type"] == "image_url":
                # Convert base64 image to proper format for Gemini
                img_type = content["image_url"]["url"].split(";")[0].split(":")[1]
                base64_data = content["image_url"]["url"].split(",")[1]

                gemini_message["parts"].append(
                    {"inline_data": {"mime_type": img_type, "data": base64_data}}
                )

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]

    return gemini_messages


# Function to stream the Gemini response
def stream_gemini_response(messages):
    response_message = ""
    gemini_messages = messages_to_gemini(messages)

    for chunk in model.generate_content(
        contents=gemini_messages,
        stream=True,
    ):
        chunk_text = chunk.text or ""
        response_message += chunk_text
        yield chunk_text

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response_message,
                }
            ],
        }
    )


def main():
    # Header
    st.header(
        "Ask AI",
    )

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"])

    # Sidebar for image upload and settings
    with st.sidebar:
        st.markdown("### 🖼️ Upload an Image")

        def add_image_to_messages():
            if "uploaded_img" in st.session_state and st.session_state.uploaded_img:
                img_file = st.session_state.uploaded_img
                img_type = img_file.type
                raw_img = Image.open(img_file)
                img = get_image_base64(raw_img)

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img}"},
                            }
                        ],
                    }
                )

                # Add a placeholder message to prompt the user for a question about the image
                if not st.session_state.get("image_uploaded_flag", False):
                    st.info("Image uploaded! Now you can ask questions about it.")
                    st.session_state.image_uploaded_flag = True

        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="uploaded_img",
            on_change=add_image_to_messages,
        )

        st.markdown("---")

        # Reset conversation button
        def reset_conversation():
            if "messages" in st.session_state:
                st.session_state.pop("messages", None)
                st.session_state.pop("image_uploaded_flag", None)

        st.button(
            "🗑️ Reset Conversation",
            on_click=reset_conversation,
        )

    # User input
    user_input = st.chat_input("Ask me anything...")

    if user_input:
        # Add user message to session
        st.session_state.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_input,
                    }
                ],
            }
        )

        # Display the new message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display AI response
        with st.chat_message("ai"):
            with st.spinner("Generating response..."):
                st.write_stream(stream_gemini_response(st.session_state.messages))

        # Reset image upload flag after handling a text question
        st.session_state.image_uploaded_flag = False


if __name__ == "__main__":
    main()
