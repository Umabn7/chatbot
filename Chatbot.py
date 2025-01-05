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
        "📚 We offer solar energy, wind energy, and waste management courses.",
        "🎓 Our courses include solar, wind energy, and waste management. Learn and grow!"
    ],
    "career_guidance": [
        "🚀 The field of renewable energy is in high demand. Would you like suggestions for certifications?",
        "🌍 Green energy careers are booming! Popular roles include sustainability analyst and energy consultant."
    ],
    "certification_help": [
        "✅ You can apply for certifications. Would you like a link?",
        "🏅 Green certifications can boost your career. Apply today!"
    ],
    "renewable_energy_advantages": [
        "🌞 Renewable energy reduces carbon emissions and helps combat climate change.",
        "💡 Lower electricity bills and promote energy independence with renewables!"
    ],
    "renewable_energy_challenges": [
        "💸 Challenges include high setup costs and weather dependency.",
        "🔑 Infrastructure and storage are key challenges in renewable energy adoption."
    ],
    "environmental_tips": [
        "🌱 Save energy by switching off appliances when not in use.",
        "♻️ Reduce waste by reusing and recycling materials wherever possible."
    ],
    "renewable_energy_trends": [
        "🌬️ Solar and wind energy are leading the global renewable energy revolution.",
        "⚡ Battery storage and green hydrogen are emerging trends in renewable energy."
    ],
    "job_opportunities": [
        "🛠️ Popular roles: Renewable energy engineer, solar technician, and energy auditor.",
        "📈 The demand for professionals in green energy fields is growing rapidly!"
    ],
}

fun_facts = [
    "💡 Did you know? The energy from the sun in one hour is enough to power the Earth for a year!",
    "🌬️ Wind turbines can reach heights taller than the Statue of Liberty!",
    "♻️ Recycling one aluminum can saves enough energy to power a TV for three hours."
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
    return random.choice(responses[intent]) if intent in responses else "🤔 Sorry, I didn't understand. Can you rephrase?"

# Streamlit app
def main():
    st.markdown("<h1 style='text-align: center; color: green;'>🌿 Green Chat Bot 🌿</h1>", unsafe_allow_html=True)
    st.subheader("Ask me about courses, careers, certifications, and more!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Your message:", key="user_input")

    if st.button("Send"):
        if user_input.strip():
            intent = predict_intent(user_input)
            response = get_response(intent)

            # Store conversation history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))

            # Display chat history with a simulated dynamic animation
            st.write("### Chat History")
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(f"**{sender}:** {message}")
                else:
                    # Simulated typing effect with ellipsis
                    placeholder = st.empty()
                    for _ in range(3):
                        placeholder.markdown(f"**{sender}:** {message[:5]}{'...' * _}")
                        st.experimental_rerun()  # Refresh Streamlit rendering
                    placeholder.markdown(f"**{sender}:** {message}")

            # Randomly share a fun fact
            if random.random() < 0.3:  # 30% chance to share a fun fact
                fun_fact = random.choice(fun_facts)
                st.markdown(f"🤓 **Fun Fact:** {fun_fact}")
        else:
            st.warning("Please enter a message to chat!")

    st.markdown("<footer style='text-align: center; margin-top: 30px;'>💡 Powered by AI | Renewable Energy Enthusiast 🌱</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
