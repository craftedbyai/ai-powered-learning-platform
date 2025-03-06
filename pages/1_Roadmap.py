import streamlit as st
import json
from typing import List, Dict, TypedDict, Any
import os
from datetime import datetime
import logging
import re
import google.generativeai as genai  # Import the google.generativeai library

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO  # Import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Define a type for a single milestone
class Milestone(TypedDict):
    title: str
    days: int
    description: str


# Define a type for the entire roadmap
class Roadmap(TypedDict):
    title: str
    total_days: int
    milestones: List[Milestone]


def clean_json_string(json_string: str) -> str:
    """
    Removes invalid control characters from a JSON string and ensures it starts and ends with curly braces.
    """
    cleaned_string = re.sub(
        r"[\x00-\x1F\x7F-\x9F]", "", json_string
    )  # Remove control characters
    cleaned_string = cleaned_string.strip()  # Remove leading/trailing whitespace

    # Ensure the string starts with a curly brace
    if not cleaned_string.startswith("{"):
        first_brace_index = cleaned_string.find("{")
        if first_brace_index != -1:
            cleaned_string = cleaned_string[first_brace_index:]
        else:
            return "{}"  # Return default valid JSON

    # Ensure the string ends with a curly brace
    if not cleaned_string.endswith("}"):
        last_brace_index = cleaned_string.rfind("}")
        if last_brace_index != -1:
            cleaned_string = cleaned_string[: last_brace_index + 1]
        else:
            return "{}"  # Return default valid JSON

    return cleaned_string


# Function to generate roadmap using Gemini-1.5-Flash:
def generate_roadmap(user_input: str, num_weeks: int, roadmap_description: str) -> str:
    """
    Generates a roadmap JSON string using Gemini-1.5-Flash,
    incorporating the number of weeks and roadmap description.
    """
    try:
        prompt = f"""
        Generate a detailed roadmap in JSON format for learning {user_input}.  The roadmap should be designed to be completed in approximately {num_weeks} weeks. {roadmap_description}

        The roadmap should include:

        - title: A concise title for the roadmap.
        - total_days: An estimated total number of days to complete the roadmap (calculated based on the number of weeks).
        - milestones: A list of milestones, each with:
            - title: A clear and specific title for the milestone.
            - days: An estimated number of days to complete the milestone.
            - description: A detailed description of what to learn in the milestone, including specific topics and concepts.

        The roadmap should be suitable for beginners with no prior knowledge.  Structure the JSON with proper nesting of milestones. Ensure valid JSON syntax. Be realistic with the total days.  Calculate total days = number of weeks * 7. Don't include any preamble or postamble, only the JSON.

        Here's an example of the expected JSON structure:
        {{
          "title": "Example Roadmap",
          "total_days": 30,
          "milestones": [
            {{
              "title": "Milestone 1",
              "days": 5,
              "description": "Learn the basics..."
            }},
            {{
              "title": "Milestone 2",
              "days": 10,
              "description": "Practice the concepts..."
            }}
          ]
        }}
        """

        logging.info(f"Prompt to Gemini: {prompt}")

        # Use the google.generativeai library and the Gemini-1.5-Flash model
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            st.error(
                "Please enter your Gemini API key in Streamlit secrets.  It should be stored as GEMINI_API_KEY."
            )
            return (
                None  # Return none if no API Key, because it will cause future errors.
            )

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(prompt)
        json_string = response.text.strip()

        # Clean the JSON string to remove invalid control characters and ensure proper start/end
        json_string = clean_json_string(json_string)

        return json_string

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        st.error(f"Error generating roadmap: {e}")
        return None


def display_roadmap(roadmap_data: str):
    """
    Displays the roadmap in Streamlit.
    """
    try:
        roadmap: Roadmap = Roadmap(**json.loads(roadmap_data))  # type: ignore
        st.header(roadmap["title"])
        st.write(f"Total Estimated Days: {roadmap['total_days']}")

        for milestone in roadmap["milestones"]:
            st.subheader(milestone["title"])
            st.write(f"Estimated Days: {milestone['days']}")
            st.write(milestone["description"])
            st.markdown("---")  # Add a separator between milestones
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {e}")
    except Exception as e:
        st.error(f"Error displaying roadmap: {e}")


def generate_pdf(roadmap_data: str) -> BytesIO:
    """Generates a PDF file from the roadmap data using ReportLab and returns it as a BytesIO object."""
    try:
        roadmap: Roadmap = Roadmap(**json.loads(roadmap_data))  # type: ignore

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add PDF metadata
        title = roadmap["title"]

        # Add content to the PDF
        story.append(Paragraph(title, styles["Title"]))
        story.append(
            Paragraph(
                f"Total Estimated Days: {roadmap['total_days']}", styles["Heading2"]
            )
        )
        story.append(Spacer(1, 0.2 * inch))

        for milestone in roadmap["milestones"]:
            story.append(Paragraph(milestone["title"], styles["Heading3"]))
            story.append(
                Paragraph(f"Estimated Days: {milestone['days']}", styles["Heading4"])
            )
            description = milestone["description"].replace("\n", "<br/>")
            story.append(Paragraph(description, styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))  # Add more space after each milestone

        # Build the PDF
        doc.build(story)
        buffer.seek(0)  # Reset buffer position
        return buffer

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        import traceback

        st.error(traceback.format_exc())
        return BytesIO()  # Return empty buffer on error


def main():
    st.title("AI Roadmap Generator")

    user_input = st.text_input("Enter the topic you want to learn:", "AI Fundamentals")
    num_weeks = st.number_input(
        "Enter the desired number of weeks for the roadmap:",
        min_value=1,
        max_value=52,
        value=4,
    )  # reasonable range
    roadmap_description = st.text_area(
        "Provide a description or specific requirements for the roadmap:",
        "Focus on practical application and hands-on projects.",
    )

    if st.button("Generate Roadmap"):
        with st.spinner("Generating roadmap..."):
            roadmap_json = generate_roadmap(user_input, num_weeks, roadmap_description)

        if roadmap_json:
            display_roadmap(roadmap_json)

            # PDF Download Option
            pdf_data = generate_pdf(roadmap_json)  # Generate PDF data here

            if pdf_data:
                st.download_button(
                    label="Download Roadmap as PDF",
                    data=pdf_data,
                    file_name=f"{user_input.replace(' ', '_')}_roadmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
            else:
                st.error("Failed to generate PDF.")


if __name__ == "__main__":
    main()
