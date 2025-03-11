import streamlit as st
import random
import json
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
import base64
import io
import time
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import PyPDF2  # For PDF processing

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
# Timer variables
if "quiz_start_time" not in st.session_state:
    st.session_state.quiz_start_time = None
if "quiz_duration" not in st.session_state:
    st.session_state.quiz_duration = 300  # Default 5 minutes in seconds
if "time_up" not in st.session_state:
    st.session_state.time_up = False
if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = ""
if "quiz_difficulty" not in st.session_state:
    st.session_state.quiz_difficulty = ""
# PDF content variable
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = ""
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = ""

# Configure Gemini API Key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")  # Store in Streamlit secrets
if not GEMINI_API_KEY:
    st.error(
        "Please enter your Gemini API key in Streamlit secrets. It should be stored as GEMINI_API_KEY."
    )
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None


# Function to generate quiz questions using Gemini
def generate_quiz(topic, difficulty, num_questions=5, pdf_content=None):
    # Modify prompt based on whether we're using a topic or PDF content
    if pdf_content:
        # Limit content length to avoid token limits
        content_preview = pdf_content[:8000]  # Limit to approximately 8000 characters
        prompt = f"""Generate a quiz based on the following document content with {num_questions} multiple-choice questions at {difficulty} difficulty level.
        Document content: {content_preview}
        
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
    else:
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
        st.info(
            f"Sending request to Gemini API for quiz generation on {'uploaded PDF' if pdf_content else f'topic: {topic}'}"
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
def generate_flashcards(topic, num_cards=10, pdf_content=None):
    # Modify prompt based on whether we're using a topic or PDF content
    if pdf_content:
        # Limit content length to avoid token limits
        content_preview = pdf_content[:8000]  # Limit to approximately 8000 characters
        prompt = f"""Generate {num_cards} flashcards based on the following document content.
        Document content: {content_preview}
        
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
    else:
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
            f"Sending request to Gemini API for flashcard generation on {'uploaded PDF' if pdf_content else f'topic: {topic}'}"
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
    st.session_state.quiz_start_time = None
    st.session_state.time_up = False
    st.session_state.quiz_topic = ""
    st.session_state.quiz_difficulty = ""
    st.session_state.pdf_content = ""
    st.session_state.pdf_filename = ""


def reset_flashcards():
    st.session_state.flashcards = None
    st.session_state.current_card = 0
    st.session_state.show_answer = False
    st.session_state.pdf_content = ""
    st.session_state.pdf_filename = ""


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


# Function to create PDF report
def create_pdf_report(quiz_data, user_answers, score, feedback_data, topic, difficulty):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=0.3 * inch,
    )

    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontSize=14, spaceAfter=0.2 * inch
    )

    normal_style = ParagraphStyle(
        "Normal", parent=styles["Normal"], fontSize=10, spaceAfter=0.1 * inch
    )

    table_cell_style = ParagraphStyle(
        "TableCell", parent=styles["Normal"], fontSize=9, leading=12
    )

    # Elements list for PDF
    elements = []

    # Add report title
    current_date = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Quiz Performance Report", title_style))
    elements.append(
        Paragraph(
            f"Topic: {topic} | Difficulty: {difficulty} | Date: {current_date}",
            styles["Italic"],
        )
    )
    elements.append(Spacer(1, 0.2 * inch))

    # Add summary
    elements.append(Paragraph("Quiz Summary", heading_style))
    elements.append(
        Paragraph(
            f"Score: {score}/{len(quiz_data)} ({int(score/len(quiz_data)*100)}%)",
            normal_style,
        )
    )
    elements.append(Spacer(1, 0.1 * inch))

    # Add performance feedback
    if feedback_data:
        elements.append(Paragraph("Performance Analysis", heading_style))
        elements.append(
            Paragraph(
                feedback_data.get("overall_performance", "No summary available"),
                normal_style,
            )
        )
        elements.append(Spacer(1, 0.1 * inch))

        # Areas for improvement
        elements.append(Paragraph("Areas for Improvement", heading_style))
        for area in feedback_data.get("improvement_areas", []):
            elements.append(Paragraph(f"• {area}", normal_style))
        elements.append(Spacer(1, 0.1 * inch))

        # Study resources
        elements.append(Paragraph("Recommended Resources", heading_style))
        for resource in feedback_data.get("study_resources", []):
            elements.append(
                Paragraph(f"<b>{resource.get('topic', 'Topic')}</b>:", normal_style)
            )
            elements.append(
                Paragraph(
                    f"• {resource.get('resource_name', 'Resource')}: {resource.get('resource_link', 'Link')}",
                    normal_style,
                )
            )
        elements.append(Spacer(1, 0.1 * inch))

        # Motivational message
        elements.append(Paragraph("Motivation", heading_style))
        elements.append(
            Paragraph(
                feedback_data.get("motivational_message", "Keep learning!"),
                normal_style,
            )
        )
        elements.append(Spacer(1, 0.2 * inch))

    # Question review table
    elements.append(Paragraph("Detailed Question Review", heading_style))

    # Create question review table with improved formatting
    table_data = [["#", "Question", "Your Answer", "Correct Answer", "Result"]]

    for i, question in enumerate(quiz_data):
        user_answer = user_answers.get(i, "Not answered")
        correct_answer = question["correct_answer"]
        result = "✓" if user_answer == correct_answer else "✗"

        # Format question text with proper wrapping
        q_text = Paragraph(question["question"], table_cell_style)
        user_ans = Paragraph(str(user_answer), table_cell_style)
        correct_ans = Paragraph(str(correct_answer), table_cell_style)

        table_data.append([str(i + 1), q_text, user_ans, correct_ans, result])

    # Create the table with adjusted column widths
    available_width = doc.width
    question_table = Table(
        table_data,
        colWidths=[
            0.05 * available_width,  # #
            0.4 * available_width,  # Question
            0.2 * available_width,  # Your Answer
            0.2 * available_width,  # Correct Answer
            0.05 * available_width,  # Result
        ],
        repeatRows=1,
    )

    question_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (4, 0), (4, -1), "CENTER"),  # Center the result column
                ("LEFTPADDING", (0, 0), (-1, -1), 4),  # Add padding for all cells
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    elements.append(question_table)

    # Footer
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(
        Paragraph("Generated by Quiz and Flashcard Generator", styles["Italic"])
    )

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


# Function to create download link for PDF
def get_pdf_download_link(pdf_bytes, filename="quiz_report.pdf"):
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href


# Timer display function
import asyncio
import streamlit as st
from datetime import datetime, timedelta


def initialize_timer():
    if "quiz_start_time" not in st.session_state:
        st.session_state.quiz_start_time = None
    if "time_up" not in st.session_state:
        st.session_state.time_up = False
    if "quiz_duration" not in st.session_state:
        st.session_state.quiz_duration = 600  # 10 minutes by default
    if "questions_answered" not in st.session_state:
        st.session_state.questions_answered = False


def setup_timer_styles():
    st.markdown(
        """
        <style>
        .timer {
            font-size: 38px !important;
            font-weight: 500 !important;
            text-align: center !important;
        }
        .timer-green {
            color: #2ecc71 !important;
        }
        .timer-orange {
            color: #f39c12 !important;
        }
        .timer-red {
            color: #e74c3c !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


async def countdown_timer(timer_placeholder):
    while True:
        if st.session_state.quiz_start_time is None or st.session_state.time_up:
            await asyncio.sleep(1)
            continue

        elapsed_time = datetime.now() - st.session_state.quiz_start_time
        remaining_time = (
            timedelta(seconds=st.session_state.quiz_duration) - elapsed_time
        )

        # Check if time is up
        if remaining_time.total_seconds() <= 0:
            st.session_state.time_up = True
            st.session_state.questions_answered = True
            timer_placeholder.markdown(
                f"""
                <p class="timer timer-red">
                    Time's Up!
                </p>
                """,
                unsafe_allow_html=True,
            )
            st.rerun()
            break

        # Format the time
        mins, secs = divmod(int(remaining_time.total_seconds()), 60)
        time_str = f"{mins:02d}:{secs:02d}"

        # Determine timer color class
        if remaining_time.total_seconds() < 60:
            color_class = "timer-red"
        elif remaining_time.total_seconds() < 180:
            color_class = "timer-orange"
        else:
            color_class = "timer-green"

        # Update the display
        timer_placeholder.markdown(
            f"""
            <p class="timer {color_class}">
                Time Remaining: {time_str}
            </p>
            """,
            unsafe_allow_html=True,
        )

        # Wait for 1 second before updating again
        await asyncio.sleep(1)


def display_timer():
    # Initialize session state variables
    initialize_timer()

    # Set up CSS styles for the timer
    setup_timer_styles()

    # Create a placeholder for the timer
    timer_placeholder = st.empty()

    # Only start the timer if the quiz has begun
    if st.session_state.quiz_start_time is not None and not st.session_state.time_up:
        # Run the async timer
        asyncio.run(countdown_timer(timer_placeholder))


# Main app layout with tabs for Quiz and Flashcards, each with topic/PDF options


# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .flashcard {
        height: 300px;
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #171717;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App header
st.title("Quiz and Flashcard Generator")

# Main tabs
main_tab = st.tabs(["Quiz Generator", "Flashcard Generator"])

with main_tab[0]:  # Quiz Generator Tab
    st.subheader("Quiz Generator")

    # Create tabs for topic-based and PDF-based quiz
    if st.session_state.quiz_data is None:
        quiz_tabs = st.tabs(["Topic-Based Quiz", "PDF-Based Quiz"])
        with quiz_tabs[0]:  # Topic-Based Quiz
            # Quiz configuration
            quiz_topic = st.text_input("Enter quiz topic:", key="quiz_topic_input")
            col1, col2 = st.columns(2)
            with col1:
                quiz_difficulty = st.selectbox(
                    "Select difficulty level:",
                    options=[
                        "Elementary",
                        "Beginner",
                        "Intermediate",
                        "Advanced",
                        "Expert",
                    ],
                    key="quiz_difficulty_slider",
                )
            with col2:
                num_questions = st.slider(
                    "Number of questions:", 3, 20, 5, key="quiz_num_questions"
                )

            # Quiz duration slider (in minutes)
            quiz_duration_min = st.slider(
                "Quiz duration (minutes):", 1, 30, 5, key="quiz_duration_min"
            )
            st.session_state.quiz_duration = (
                quiz_duration_min * 60
            )  # Convert to seconds

            # Generate quiz button
            if st.button("Generate Quiz", key="generate_topic_quiz"):
                if quiz_topic:
                    with st.spinner(
                        f"Generating {quiz_difficulty} level quiz about {quiz_topic}..."
                    ):
                        st.session_state.quiz_data = generate_quiz(
                            quiz_topic, quiz_difficulty, num_questions
                        )
                        if st.session_state.quiz_data:
                            st.session_state.quiz_topic = quiz_topic
                            st.session_state.quiz_difficulty = quiz_difficulty
                            st.session_state.quiz_start_time = datetime.now()
                            st.success("Quiz generated successfully!")
                            st.rerun()
                else:
                    st.warning("Please enter a quiz topic.")

        with quiz_tabs[1]:  # PDF-Based Quiz
            st.markdown("### Upload a PDF Document")
            uploaded_file = st.file_uploader(
                "Choose a PDF file", type=["pdf"], key="quiz_pdf_uploader"
            )

            if uploaded_file is not None:
                # Extract text from the uploaded PDF
                with st.spinner("Extracting text from PDF..."):
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if pdf_text:
                        st.session_state.pdf_content = pdf_text
                        st.session_state.pdf_filename = uploaded_file.name
                        st.success(
                            f"Successfully extracted text from {uploaded_file.name}"
                        )
                        # Show a small preview of the extracted text
                        with st.expander("PDF Content Preview"):
                            st.text(
                                pdf_text[:500] + "..."
                                if len(pdf_text) > 500
                                else pdf_text
                            )
                    else:
                        st.error(
                            "Failed to extract text from the PDF. Please try another file."
                        )

            # Only show quiz configuration if PDF content is available
            if st.session_state.pdf_content:
                st.markdown("### Configure PDF-Based Quiz")
                col1, col2 = st.columns(2)
                with col1:
                    pdf_quiz_difficulty = st.selectbox(
                        "Select difficulty level:",
                        options=[
                            "Elementary",
                            "Beginner",
                            "Intermediate",
                            "Advanced",
                            "Expert",
                        ],
                        key="pdf_quiz_difficulty_slider",
                    )
                with col2:
                    pdf_num_questions = st.slider(
                        "Number of questions:", 3, 20, 5, key="pdf_num_questions"
                    )

                # Quiz duration slider (in minutes)
                pdf_quiz_duration_min = st.slider(
                    "Quiz duration (minutes):", 1, 30, 5, key="pdf_quiz_duration_min"
                )

                # Generate quiz button
                if st.button("Generate Quiz from PDF", key="generate_pdf_quiz"):
                    with st.spinner(
                        f"Generating {pdf_quiz_difficulty} level quiz from PDF..."
                    ):
                        st.session_state.quiz_data = generate_quiz(
                            "PDF Content",
                            pdf_quiz_difficulty,
                            pdf_num_questions,
                            st.session_state.pdf_content,
                        )
                        if st.session_state.quiz_data:
                            st.session_state.quiz_topic = (
                                f"PDF: {st.session_state.pdf_filename}"
                            )
                            st.session_state.quiz_difficulty = pdf_quiz_difficulty
                            st.session_state.quiz_duration = (
                                pdf_quiz_duration_min * 60
                            )  # Convert to seconds
                            st.session_state.quiz_start_time = datetime.now()
                            st.success("Quiz generated successfully!")
                            st.rerun()

    # Show results if all questions answered or time's up
    elif st.session_state.questions_answered or st.session_state.time_up:
        # Calculate score
        correct_answers = 0
        total_questions = len(st.session_state.quiz_data)
        score = st.session_state.score
        progress = score / total_questions

        if st.session_state.time_up:
            st.warning("⏰ Time's up! Your quiz has ended.")

        for i, q in enumerate(st.session_state.quiz_data):
            if (
                i in st.session_state.user_answers
                and st.session_state.user_answers[i] == q["correct_answer"]
            ):
                correct_answers += 1

        st.session_state.score = correct_answers

        # Show results
        st.subheader("Quiz Results")
        st.markdown(
            f"**Score:** {correct_answers}/{len(st.session_state.quiz_data)} ({int(correct_answers/len(st.session_state.quiz_data)*100)}%)"
        )
        st.progress(progress)
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

        # Generate personalized feedback
        if st.button("Generate Detailed Feedback"):
            with st.spinner("Generating personalized feedback..."):
                feedback = generate_personalized_feedback(
                    st.session_state.quiz_data,
                    st.session_state.user_answers,
                    st.session_state.score,
                )

                if feedback:
                    st.markdown("### Performance Analysis")
                    st.write(feedback.get("overall_performance", ""))

                    st.markdown("### Areas for Improvement")
                    for area in feedback.get("improvement_areas", []):
                        st.markdown(f"- {area}")

                    st.markdown("### Recommended Resources")
                    for resource in feedback.get("study_resources", []):
                        st.markdown(f"**{resource.get('topic', '')}**")
                        st.markdown(
                            f"- {resource.get('resource_name', '')}: {resource.get('resource_link', '')}"
                        )

                    st.markdown("### Motivation")
                    st.write(feedback.get("motivational_message", ""))

                    # Generate PDF report
                    pdf_buffer = create_pdf_report(
                        st.session_state.quiz_data,
                        st.session_state.user_answers,
                        st.session_state.score,
                        feedback,
                        st.session_state.quiz_topic,
                        st.session_state.quiz_difficulty,
                    )

                    # Provide download link
                    st.markdown("### Download Your Report")
                    st.markdown(
                        get_pdf_download_link(pdf_buffer, "quiz_report.pdf"),
                        unsafe_allow_html=True,
                    )

        # Review questions
        st.subheader("Question Review")
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
        # Reset quiz button
        if st.button("Start New Quiz"):
            reset_quiz()
            st.rerun()

    else:
        st.subheader("Take the Quiz")

        # Display current question
        current_q = st.session_state.current_question
        question = st.session_state.quiz_data[current_q]

        st.markdown(
            f"### Question {current_q + 1} of {len(st.session_state.quiz_data)}"
        )
        st.markdown(f"**{question['question']}**")

        # Get user's answer
        answer = st.radio(
            "Select your answer:",
            question["options"],
            key=f"q_{current_q}",
            index=None,
        )

        # Store the answer
        st.session_state.user_answers[current_q] = answer

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if current_q > 0:
                if st.button("Previous Question", key="prev_q"):
                    prev_question()
                    st.rerun()

        with col2:
            if current_q < len(st.session_state.quiz_data) - 1:
                if st.button("Next Question", key="next_q"):
                    next_question()
                    st.rerun()
            else:
                if st.button("Finish Quiz", key="finish_q"):
                    st.session_state.questions_answered = True
                    st.rerun()

        display_timer()


with main_tab[1]:  # Flashcard Generator Tab
    st.subheader("Flashcard Generator")

    # Create tabs for topic-based and PDF-based flashcards
    if st.session_state.flashcards is None:
        flashcard_tabs = st.tabs(["Topic-Based Flashcards", "PDF-Based Flashcards"])
        with flashcard_tabs[0]:  # Topic-Based Flashcards
            fc_topic = st.text_input("Enter flashcard topic:", key="fc_topic_input")
            num_cards = st.slider(
                "Number of flashcards:", 5, 30, 10, key="num_cards_slider"
            )

            if st.button("Generate Flashcards", key="gen_topic_fc_btn"):
                if fc_topic:
                    with st.spinner(f"Generating flashcards about {fc_topic}..."):
                        st.session_state.flashcards = generate_flashcards(
                            fc_topic, num_cards
                        )
                        if st.session_state.flashcards:
                            st.success("Flashcards generated successfully!")
                            st.rerun()
                else:
                    st.warning("Please enter a flashcard topic.")

        with flashcard_tabs[1]:  # PDF-Based Flashcards
            st.markdown("### Upload a PDF Document")
            fc_uploaded_file = st.file_uploader(
                "Choose a PDF file", type=["pdf"], key="fc_pdf_uploader"
            )

            if fc_uploaded_file is not None:
                # Extract text from the uploaded PDF
                with st.spinner("Extracting text from PDF..."):
                    fc_pdf_text = extract_text_from_pdf(fc_uploaded_file)
                    if fc_pdf_text:
                        st.session_state.pdf_content = fc_pdf_text
                        st.session_state.pdf_filename = fc_uploaded_file.name
                        st.success(
                            f"Successfully extracted text from {fc_uploaded_file.name}"
                        )
                        # Show a small preview of the extracted text
                        with st.expander("PDF Content Preview"):
                            st.text(
                                fc_pdf_text[:500] + "..."
                                if len(fc_pdf_text) > 500
                                else fc_pdf_text
                            )
                    else:
                        st.error(
                            "Failed to extract text from the PDF. Please try another file."
                        )

            # Only show flashcard configuration if PDF content is available
            if st.session_state.pdf_content:
                st.markdown("### Configure PDF-Based Flashcards")
                pdf_num_cards = st.slider(
                    "Number of flashcards:", 5, 30, 10, key="pdf_num_cards_slider"
                )

                if st.button("Generate Flashcards from PDF", key="gen_pdf_fc_btn"):
                    with st.spinner("Generating flashcards from PDF..."):
                        st.session_state.flashcards = generate_flashcards(
                            f"PDF: {st.session_state.pdf_filename}",
                            pdf_num_cards,
                            st.session_state.pdf_content,
                        )
                        if st.session_state.flashcards:
                            st.success("Flashcards generated successfully!")
                            st.rerun()

    # Display flashcards if available
    else:
        st.subheader("Study Flashcards")

        # Display current flashcard
        current_card = st.session_state.current_card
        card = st.session_state.flashcards[current_card]

        # Flashcard display
        st.markdown(
            f"#### Card {current_card + 1} of {len(st.session_state.flashcards)}"
        )

        # Card content with toggle
        if st.session_state.show_answer:
            st.markdown(
                f"""
                <div class="flashcard">
                    <div style="font-size: 1.2rem; color: #666;">Question:</div>
                    <div style="margin-bottom: 1rem;">{card["front"]}</div>
                    <div style="font-size: 1.2rem; color: #666;">Answer:</div>
                    <div>{card["back"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="flashcard">
                    <div style="font-size: 1.2rem; color: #666;">Question:</div>
                    <div>{card["front"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Flashcard navigation buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            if current_card > 0:
                if st.button("⬅️ Previous", key="prev_card"):
                    prev_card()
                    st.rerun()

        with col2:
            flip_text = (
                "Show Answer" if not st.session_state.show_answer else "Hide Answer"
            )
            if st.button(flip_text, key="flip_card"):
                flip_card()
                st.rerun()

        with col3:
            if current_card < len(st.session_state.flashcards) - 1:
                if st.button("Next ➡️", key="next_card"):
                    next_card()
                    st.rerun()

        with col4:
            if st.button("Reset Flashcards", key="reset_fc"):
                reset_flashcards()
                st.rerun()

# Footer
st.markdown("---")
st.markdown("### 📚 Quiz & Flashcard Generator")
st.markdown("This app uses Google's Gemini API to generate quizzes and flashcards.")
