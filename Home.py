import streamlit as st


def main():
    st.set_page_config(
        page_title="Smart AI Learning Assistant", page_icon="🎓", layout="wide"
    )

    st.title("Welcome to Smart AI Learning Assistant")
    st.write("Empowering learning with cutting-edge AI technologies.")
    st.markdown("---")
    st.markdown("## Our Services")
    st.write(
        "Explore the powerful features we offer to enhance your learning experience:"
    )

    services = {
        "Roadmap": {
            "description": "Generate personalized learning roadmaps based on your goals and interests.",
            "link": "/Roadmap",
        },
        "Course Recommendation": {
            "description": "Get personalized course recommendations tailored to your goals and interests.",
            "link": "/Course_Recommendation",
        },
        "Chat with PDF": {
            "description": "Upload your PDF documents and interact with the content using AI.",
            "link": "/Chat_with_Pdf",
        },
        "Quiz and Flashcards": {
            "description": "Create and practice with AI-generated quizzes and flashcards on various topics.",
            "link": "/Quiz_Flashcards",
        },
        "Ask AI": {
            "description": "Engage with our AI assistant for instant learning support and Q&A.",
            "link": "/Ask_Ai",
        },
    }

    for service, details in services.items():
        st.subheader(service)
        st.write(details["description"])
        st.write(f"[Explore {service}]({details['link']})")

    st.markdown("---")
    # st.write("Ready to explore?")
    # if st.button("Explore All Services"):
    #     st.write("Navigate to the service links above to start using our platform!")


if __name__ == "__main__":
    main()