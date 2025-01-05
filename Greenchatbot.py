import random
import streamlit as st
import spacy
from spacy.training import Example

# Data for training
training_data = [
    ("Tell me about your courses", "course_info"),
    ("What training programs do you offer?", "course_info"),
    ("I want career guidance", "career_guidance"),
    ("What are the job opportunities?", "job_opportunities"),
    ("How can I get certified?", "certification_help"),
    ("What are the benefits of renewable energy?", "renewable_energy_advantages"),
    ("What are the challenges of renewable energy?", "renewable_energy_challenges"),
    ("Can you give me environmental tips?", "environmental_tips"),
    ("What are the latest trends in renewable energy?", "renewable_energy_trends")
]

responses = {
    "course_info": [
        "We offer solar energy, wind energy, and waste management courses.",
        "Our courses include solar, wind energy, and waste management."
    ],
    "career_guidance": [
        "The field of renewable energy is in high demand. Would you like any suggestions for certifications?",
        "Green energy careers are booming nowadays, and roles like sustainability analyst and energy consultant are popular!"
    ],
    "certification_help": [
        "You can apply for certifications. Would you like a link?",
        "Green Certifications can boost your career. Apply and learn more!"
    ],
    "renewable_energy_advantages": [
        "Renewable energy reduces carbon emissions and helps combat climate change.",
        "Using renewable energy can lower your electricity bills and promote energy independence."
    ],
    "renewable_energy_challenges": [
        "Some challenges include high initial setup costs and dependency on weather conditions.",
        "Infrastructure and storage solutions are key challenges in renewable energy adoption."
    ],
    "environmental_tips": [
        "Save energy by switching off appliances when not in use.",
        "Reduce waste by reusing and recycling materials wherever possible."
    ],
    "renewable_energy_trends": [
        "Solar and wind energy are leading the global renewable energy revolution.",
        "Battery storage and green hydrogen are emerging trends in renewable energy."
    ],
    "job_opportunities": [
        "Popular roles include renewable energy engineer, solar technician, and energy auditor.",
        "The demand for professionals in green energy fields is growing rapidly worldwide."
    ]
}

# Train an NLP model using spaCy
def train_spacy_model():
    nlp = spacy.blank("en")  # Create a blank English NLP model
    textcat = nlp.add_pipe("textcat", last=True)  # Add a Text Categorizer

    # Add labels to the text categorizer
    for _, label in training_data:
        textcat.add_label(label)

    # Training
    optimizer = nlp.begin_training()
    for i in range(10):  # Number of epochs
        random.shuffle(training_data)
        losses = {}
        for text, label in training_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, {"cats": {label: 1.0}})
            nlp.update([example], losses=losses, drop=0.2, sgd=optimizer)
    return nlp

# Load the trained model
nlp_model = train_spacy_model()

# Predict the intent of user input
def predict_intent(text):
    doc = nlp_model(text)
    predicted_label = max(doc.cats, key=doc.cats.get)
    return predicted_label

# Get a response based on the intent
def get_response(intent):
    return random.choice(responses[intent]) if intent in responses else "Sorry, I didn't understand. Can you rephrase?"

# Streamlit app
def main():
    st.title("Green Chat Bot with NLP")
    st.subheader("Ask me about courses, career guidance, certifications, and more!")

    user_input = st.text_input("Your message:")

    if st.button("Send"):
        if user_input.strip():
            intent = predict_intent(user_input)
            response = get_response(intent)
            st.text_area("Chatbot Response:", response, height=100)
        else:
            st.warning("Please enter a message to chat!")

if __name__ == "__main__":
    main()
