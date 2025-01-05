import random
import spacy
from spacy.training import Example
import streamlit as st

# Training data
training_data = [
    ("Tell me about your courses", "course_info"),
    ("What training programs do you offer?", "course_info"),
    ("I want career guidance", "career_guidance"),
    ("What are the job opportunities?", "job_opportunities"),
    ("How can I get certified?", "certification_help"),
    ("What are the benefits of renewable energy?", "renewable_energy_advantages"),
    ("What are the challenges of renewable energy?", "renewable_energy_challenges"),
    ("Can you give me environmental tips?", "environmental_tips"),
    ("What are the latest trends in renewable energy?", "renewable_energy_trends"),
]

responses = {
    "course_info": [
        "We offer courses on solar energy, wind energy, and waste management.",
        "Our training programs include solar energy, wind energy, and waste management. Learn and grow!"
    ],
    "career_guidance": [
        "The renewable energy field is booming! Would you like suggestions for certifications or job roles?",
        "Green energy careers are in high demand. Roles like sustainability analyst and energy consultant are popular!"
    ],
    "certification_help": [
        "You can apply for certifications in renewable energy. Would you like a link?",
        "Green certifications are great for boosting your career. Learn more and get certified!"
    ],
    "renewable_energy_advantages": [
        "Renewable energy helps combat climate change by reducing carbon emissions.",
        "Using renewable energy can lower your electricity bills and promote energy independence."
    ],
    "renewable_energy_challenges": [
        "Challenges include high initial setup costs and weather dependency.",
        "Infrastructure and energy storage solutions remain critical challenges in renewable energy adoption."
    ],
    "environmental_tips": [
        "Switch off appliances when not in use to save energy.",
        "Reuse and recycle materials wherever possible to reduce waste."
    ],
    "renewable_energy_trends": [
        "Solar and wind energy are driving the renewable energy revolution.",
        "Battery storage and green hydrogen are emerging as the next big trends in renewable energy."
    ],
    "job_opportunities": [
        "Popular roles include renewable energy engineer, solar technician, and energy auditor.",
        "The demand for skilled professionals in renewable energy is rapidly growing worldwide."
    ],
}

fun_facts = [
    "Did you know? The energy from the sun in one hour is enough to power the Earth for a year!",
    "Wind turbines can reach heights taller than the Statue of Liberty!",
    "Recycling one aluminum can saves enough energy to power a TV for three hours.",
    "Hydropower is the oldest form of renewable energy, dating back to ancient Greece!"
]

# Train spaCy model
def train_spacy_model():
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat", last=True)
    for _, label in training_data:
        textcat.add_label(label)
    optimizer = nlp.begin_training()
    for _ in range(10):
        random.shuffle(training_data)
        losses = {}
        for text, label in training_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, {"cats": {label: 1.0}})
            nlp.update([example], losses=losses, drop=0.2, sgd=optimizer)
    return nlp

# Load model
nlp_model = train_spacy_model()

# Predict intent
def predict_intent(text):
    doc = nlp_model(text)
    predicted_label = max(doc.cats, key=doc.cats.get)
    return predicted_label

# Get response
def get_response(intent):
    if intent in responses:
        bot_response = random.choice(responses[intent])
        # Add a fun fact after every response
        fun_fact = random.choice(fun_facts)
        return f"{bot_response}\n\nFun Fact: {fun_fact}"
    return "I'm sorry, I didn't quite catch that. Could you rephrase?"

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "Chat History", "About"])

    if app_mode == "Home":
        st.title("🌿 Green Chat Bot 🌿")
        st.subheader("Ask me about renewable energy, certifications, careers, and more!")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Your message:")

        if st.button("Send"):
            if user_input.strip():
                intent = predict_intent(user_input)
                response = get_response(intent)

                # Save to chat history
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", response))

                st.write("### Bot Response")
                st.markdown(f"**Bot:** {response}")
            else:
                st.warning("Please enter a message to chat!")

    elif app_mode == "Chat History":
        st.title("📜 Chat History")
        if "chat_history" in st.session_state and st.session_state.chat_history:
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(f"**{sender}:** {message}")
                else:
                    st.markdown(f"**{sender}:** {message}")
        else:
            st.info("No chat history found. Start a conversation on the Home page!")

    elif app_mode == "About":
        st.title("About Green Chat Bot")
        st.write("""
        🌱 **Green Chat Bot** is an interactive assistant designed to help you learn about renewable energy, certifications, 
        job opportunities, and more. 

        🤖 Features:
        - Answer queries about renewable energy.
        - Provide career guidance.
        - Share fun facts about sustainability.

        Built with **Streamlit** and **spaCy**.
        """)

if __name__ == "__main__":
    main()
