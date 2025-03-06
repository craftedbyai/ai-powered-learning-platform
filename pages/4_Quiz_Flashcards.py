import streamlit as st
import random
import json
import requests
from datetime import datetime
import google.generativeai as genai
import base64
import io

# Initialize session state variables if they don't exist
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "questions_answered" not in st.session_state:
    st.session_state.questions_answered = False
if "flashcards" not in st.session_state:
    st.session_state.flashcards = None
if "current_card" not in st.session_state:
    st.session_state.current_card = 0
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False

# Configure Gemini API Key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")  # Store in Streamlit secrets
if not GEMINI_API_KEY:
    st.error(
        "Please enter your Gemini API key in Streamlit secrets.  It should be stored as GEMINI_API_KEY."
    )
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")


# Function to generate quiz questions using Gemini
def generate_quiz(topic, difficulty, num_questions=5):
    prompt = f"""Generate a quiz about {topic} with {num_questions} multiple-choice questions at {difficulty} difficulty level.
    Return the response as a valid JSON array in the following format:
    [
        {{
            "question": "Question text here",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Correct option here (exactly matching one of the options)",
            "explanation": "Brief explanation of why this is the correct answer"
        }},
        ...
    ]
    Make sure the JSON is valid and properly formatted. ONLY return the JSON, no additional text before or after.
    """

    try:
        # Debug information
        st.info(f"Sending request to Gemini API for quiz generation on topic: {topic}")

        response = model.generate_content(prompt)

        # Check if response is successful
        if response.prompt_feedback:
            if response.prompt_feedback.block_reason:
                st.error(
                    f"Gemini API blocked response. Reason: {response.prompt_feedback.block_reason}"
                )
                st.error(
                    f"Blocked Reason Message: {response.prompt_feedback.safety_ratings}"
                )  # Display more information
                return None
            elif response.prompt_feedback.safety_ratings:
                for rating in response.prompt_feedback.safety_ratings:
                    if rating.probability > 0.5 and rating.severity not in [
                        "HARM_PROBABILITY_LOW",
                        "HARM_PROBABILITY_NEGLIGIBLE",
                    ]:  # Check that ratings are not negligible
                        st.warning(
                            f"Gemini API returned a safety rating issue for category: {rating.category}, severity: {rating.severity}, and probability: {rating.probability}"
                        )

        response_text = response.text.strip()

        # # Debug: Show the raw response text
        # st.write("Raw response from Gemini (first 300 chars):")
        # st.text(
        #     response_text[:300] + "..." if len(response_text) > 300 else response_text
        # )

        # Find JSON content between brackets
        start_idx = response_text.find("[")
        end_idx = response_text.rfind("]") + 1

        if start_idx != -1 and end_idx != -1:
            json_content = response_text[start_idx:end_idx]
            try:
                quiz_data = json.loads(json_content)
                return quiz_data
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in response: {str(e)}")
                st.code(json_content)
                return None
        else:
            st.error("Failed to extract JSON data from the response.")
            return None

    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        st.exception(e)
        return None


# Function to generate flashcards using Gemini
def generate_flashcards(topic, num_cards=10):
    prompt = f"""Generate {num_cards} flashcards about {topic}.
    Return the response as a valid JSON array in the following format:
    [
        {{
            "front": "Term or question here",
            "back": "Definition or answer here"
        }},
        ...
    ]
    Make sure the JSON is valid and properly formatted. ONLY return the JSON, no additional text before or after.
    """

    try:
        # Debug information
        st.info(
            f"Sending request to Gemini API for flashcard generation on topic: {topic}"
        )

        response = model.generate_content(prompt)

        # Check if response is successful
        if response.prompt_feedback:
            if response.prompt_feedback.block_reason:
                st.error(
                    f"Gemini API blocked response. Reason: {response.prompt_feedback.block_reason}"
                )
                st.error(
                    f"Blocked Reason Message: {response.prompt_feedback.safety_ratings}"
                )  # Display more information
                return None
            elif response.prompt_feedback.safety_ratings:
                for rating in response.prompt_feedback.safety_ratings:
                    if rating.probability > 0.5 and rating.severity not in [
                        "HARM_PROBABILITY_LOW",
                        "HARM_PROBABILITY_NEGLIGIBLE",
                    ]:  # Check that ratings are not negligible
                        st.warning(
                            f"Gemini API returned a safety rating issue for category: {rating.category}, severity: {rating.severity}, and probability: {rating.probability}"
                        )

        response_text = response.text.strip()

        # # Debug: Show the raw response text
        # st.write("Raw response from Gemini (first 300 chars):")
        # st.text(
        #     response_text[:300] + "..." if len(response_text) > 300 else response_text
        # )

        # Try direct JSON parsing first
        try:
            flashcards_data = json.loads(response_text)
            return flashcards_data
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON content between brackets
            try:
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]") + 1

                if start_idx != -1 and end_idx != -1:
                    json_content = response_text[start_idx:end_idx]
                    flashcards_data = json.loads(json_content)
                    return flashcards_data
                else:
                    st.error("Failed to extract JSON data from the response.")
                    return None
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in response: {str(e)}")
                st.code(response_text)
                return None

    except Exception as e:
        st.error(f"Error generating flashcards: {str(e)}")
        st.exception(e)
        return None


# Navigation functions
def next_question():
    if st.session_state.current_question < len(st.session_state.quiz_data) - 1:
        st.session_state.current_question += 1
    else:
        st.session_state.questions_answered = True


def prev_question():
    if st.session_state.current_question > 0:
        st.session_state.current_question -= 1


def next_card():
    st.session_state.show_answer = False
    if st.session_state.current_card < len(st.session_state.flashcards) - 1:
        st.session_state.current_card += 1


def prev_card():
    st.session_state.show_answer = False
    if st.session_state.current_card > 0:
        st.session_state.current_card -= 1


def flip_card():
    st.session_state.show_answer = not st.session_state.show_answer


def reset_quiz():
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.quiz_data = None
    st.session_state.user_answers = {}
    st.session_state.questions_answered = False


def reset_flashcards():
    st.session_state.flashcards = None
    st.session_state.current_card = 0
    st.session_state.show_answer = False


def generate_personalized_feedback(quiz_data, user_answers, score):
    # Prepare a prompt for Gemini to generate personalized feedback
    incorrect_questions = []
    topic_difficulty = {}

    for i, question in enumerate(quiz_data):
        user_answer = user_answers.get(i, None)

        # Track difficulty of incorrect questions
        if user_answer != question["correct_answer"]:
            incorrect_questions.append(
                {
                    "question": question["question"],
                    "user_answer": user_answer,
                    "correct_answer": question["correct_answer"],
                    "explanation": question["explanation"],
                }
            )

    # Construct a detailed prompt for feedback
    prompt = f"""Provide a concise and practical learning assessment based on the following quiz performance:

Total Questions: {len(quiz_data)}
Correct Answers: {score}
Incorrect Questions: {len(incorrect_questions)}

Incorrect Questions Details:
{json.dumps(incorrect_questions, indent=2)}

Please generate a focused feedback report that includes:
1. Brief performance summary
2. Key areas of improvement
3. 2-3 specific learning resources or study strategies (that can include books, youtube playlists, blogs, online courses, etc)
4. A short, motivational message

Format the response as a JSON with the following structure:
{{
    "overall_performance": "",
    "improvement_areas": [],
    "study_resources": [
        {{
            "topic": "",
            "resource_name": "",
            "resource_link": ""
        }}
    ],
    "motivational_message": ""
}}
"""

    try:
        response = model.generate_content(prompt)

        # Extract and parse the JSON response
        response_text = response.text.strip()

        # Find JSON content between brackets
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx != -1 and end_idx != -1:
            json_content = response_text[start_idx:end_idx]
            try:
                feedback_data = json.loads(json_content)
                return feedback_data
            except json.JSONDecodeError as e:
                st.error(f"Error parsing feedback JSON: {str(e)}")
                return None
        else:
            st.error("Failed to extract JSON data from the feedback response.")
            return None

    except Exception as e:
        st.error(f"Error generating personalized feedback: {str(e)}")
        return None


# Main app title
st.title("Quiz and Flashcard Generator")


# Create tabs for quiz and flashcards
tab1, tab2 = st.tabs(["Quiz", "Flashcards"])

# Quiz tab
with tab1:
    if st.session_state.quiz_data is None:
        st.subheader("Generate a Quiz")
        topic = st.text_input("Enter the topic for your quiz:")

        col1, col2 = st.columns(2)
        with col1:
            difficulty = st.selectbox(
                "Select difficulty level:", ["Easy", "Medium", "Hard"]
            )
        with col2:
            num_questions = st.slider("Number of questions:", 3, 10, 5)

        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz questions..."):
                st.session_state.quiz_data = generate_quiz(
                    topic, difficulty, num_questions
                )
                if st.session_state.quiz_data:
                    st.success("Quiz generated successfully!")
                    st.rerun()

    elif st.session_state.questions_answered:
        # Show quiz results
        total_questions = len(st.session_state.quiz_data)
        score = st.session_state.score
        progress = score / total_questions

        st.subheader("Quiz Results")
        st.progress(progress)
        st.write(
            f"Your score: {st.session_state.score}/{len(st.session_state.quiz_data)}"
        )

        # Display performance message
        if progress == 1.0:
            st.success("Perfect score! Excellent work! 🎉")
        elif progress >= 0.8:
            st.success("Great job! 👏")
        elif progress >= 0.6:
            st.info("Good effort! 👍")
        elif progress >= 0.4:
            st.warning("Keep practicing! 💪")
        else:
            st.error("You might need to review this topic more. 📚")
        st.subheader("Detailed Question Review")

        for i, question in enumerate(st.session_state.quiz_data):
            with st.expander(f"Question {i+1}: {question['question']}"):
                user_answer = st.session_state.user_answers.get(i, None)
                correct_answer = question["correct_answer"]

                # Show if the answer was correct
                if user_answer == correct_answer:
                    st.success(f"Your answer: {user_answer} ✓")
                else:
                    st.error(f"Your answer: {user_answer or 'Not answered'} ✗")
                    st.success(f"Correct answer: {correct_answer}")

                # Show explanation
                st.info(f"Explanation: {question['explanation']}")

        if st.button("Take Another Quiz"):
            reset_quiz()
            st.rerun()

        with st.spinner("Analyzing your performance..."):
            personalized_feedback = generate_personalized_feedback(
                st.session_state.quiz_data,
                st.session_state.user_answers,
                st.session_state.score,
            )

        # Personalized Feedback Section
        if personalized_feedback:
            st.subheader("Personalized Insights")

            # Performance Summary
            st.markdown(
                f"""
            ### 📊 Performance Overview
            {personalized_feedback.get('overall_performance', 'No summary available')}
            """
            )

            # Areas for Improvement
            st.markdown("### 🎯 Key Improvement Areas")
            for area in personalized_feedback.get("improvement_areas", []):
                st.markdown(f"- {area}")

            # Study Resources
            st.markdown("### 📚 Recommended Resources")
            for resource in personalized_feedback.get("study_resources", []):
                st.markdown(
                    f"""
                #### {resource.get('topic', 'Unknown Topic')}
                - **Resource:** [{resource.get('resource_name', 'Link')}]({resource.get('resource_link', '#')})
                """
                )

            # Motivational Message
            st.markdown(
                f"""
            ### 💡 Motivation
            *{personalized_feedback.get('motivational_message', 'Keep learning and improving!')}*
            """
            )

    else:
        # Display current question
        current = st.session_state.current_question
        question_data = st.session_state.quiz_data[current]

        # Progress indicator
        progress_text = f"Question {current + 1} of {len(st.session_state.quiz_data)}"
        st.progress((current) / len(st.session_state.quiz_data))
        st.write(progress_text)

        # Question
        st.subheader(question_data["question"])

        # Answer selection
        selected_option = st.radio(
            "Select your answer:",
            question_data["options"],
            key=f"q{current}",
            index=None,
        )

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if current > 0:
                if st.button("Previous", key=f"prev_{current}"):
                    prev_question()
                    st.rerun()

        with col2:
            if selected_option:
                if st.button("Submit Answer", key=f"submit_{current}"):
                    st.session_state.user_answers[current] = selected_option
                    if selected_option == question_data["correct_answer"]:
                        st.session_state.score += 1
                    next_question()
                    st.rerun()

        with col3:
            if current < len(st.session_state.quiz_data) - 1:
                if st.button("Skip", key=f"skip_{current}"):
                    next_question()
                    st.rerun()

# Flashcards tab
with tab2:
    if st.session_state.flashcards is None:
        st.subheader("Generate Flashcards")
        topic = st.text_input("Enter the topic for your flashcards:")
        num_cards = st.slider("Number of flashcards:", 5, 20, 10)

        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                st.session_state.flashcards = generate_flashcards(topic, num_cards)
                if st.session_state.flashcards:
                    st.success("Flashcards generated successfully!")
                    st.rerun()

    else:
        # Display flashcards
        current = st.session_state.current_card
        total_cards = len(st.session_state.flashcards)

        # Progress indicator
        progress_text = f"Card {current + 1} of {total_cards}"
        st.progress((current) / total_cards)
        st.write(progress_text)

        # Flashcard container with some styling
        card_container = st.container()
        with card_container:
            # Create a card-like appearance
            card = st.container()
            card.markdown(
                f"""
                <div style="
                    padding: 20px;
                    border-radius: 10px;
                    background-color: {'#171717' if not st.session_state.show_answer else '#000000'};
                    min-height: 200px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin: 20px 0;
                    transition: all 0.3s ease;
                ">
                    <h3>{st.session_state.flashcards[current]['front'] if not st.session_state.show_answer else st.session_state.flashcards[current]['back']}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Flip button
        if st.button("Flip Card"):
            flip_card()
            st.rerun()

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if current > 0:
                if st.button("Previous Card"):
                    prev_card()
                    st.rerun()

        with col2:
            if st.button("Reset Flashcards"):
                reset_flashcards()
                st.rerun()

        with col3:
            if current < total_cards - 1:
                if st.button("Next Card"):
                    next_card()
                    st.rerun()

# Footer
st.markdown("---")
# st.caption(f"© {datetime.now().year} Quiz and Flashcard Generator | Powered by Streamlit and Gemini")
