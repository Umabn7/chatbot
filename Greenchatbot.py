import random
import spacy
from spacy.training import Example
import streamlit as st

# Enhanced training data with trivia/quiz questions
training_data = [
    ("Tell me about your courses", "course_info"),
    ("What training programs do you offer?", "course_info"),
    ("Can you tell me about your courses?", "course_info"),
    ("What are your courses in renewable energy?", "course_info"),
    ("I want to learn about solar energy courses", "course_info"),
    ("What courses do you have for wind energy?", "course_info"),
    ("I need career guidance in renewable energy", "career_guidance"),
    ("What job opportunities are available in renewable energy?", "job_opportunities"),
    ("How can I get certified in renewable energy?", "certification_help"),
    ("What certifications do you offer?", "certification_help"),
    ("Tell me the benefits of renewable energy", "renewable_energy_advantages"),
    ("What are the main benefits of renewable energy?", "renewable_energy_advantages"),
    ("What are the challenges of renewable energy?", "renewable_energy_challenges"),
    ("Can you list some environmental tips?", "environmental_tips"),
    ("What are the latest trends in renewable energy?", "renewable_energy_trends"),
    ("Can you help me with my career path?", "career_guidance"),
    ("How do I pursue a career in renewable energy?", "career_guidance"),
    ("Hello", "greeting"),
    ("Hi", "greeting"),
    ("Good morning", "greeting"),
    ("Good evening", "greeting"),
    ("Hey", "greeting"),
    ("How are you?", "greeting"),
    ("What's up?", "greeting"),
    ("Quiz me on renewable energy", "quiz_request"),
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
    "greeting": [
        "Hello there! How can I assist you today?",
        "Hi! How can I help you with your renewable energy questions?",
        "Good day! What can I do for you?",
        "Hey! What would you like to know about renewable energy?"
    ],
    "quiz_request": [
        "Sure! Here's your trivia question about renewable energy:",
        "Let's test your knowledge on renewable energy!"
    ],
}

fun_facts = [
    "Did you know? The energy from the sun in one hour is enough to power the Earth for a year!",
    "Wind turbines can reach heights taller than the Statue of Liberty!",
    "Recycling one aluminum can saves enough energy to power a TV for three hours.",
    "Hydropower is the oldest form of renewable energy, dating back to ancient Greece!",
    "Solar energy can be harnessed anywhere the sun shines, from deserts to rooftops!",
    "By 2050, renewable energy could power the entire world if we make the switch!",
    "The world's largest solar park, the Bhadla Solar Park in India, can generate 2,245 MW!",
]

# Trivia/quiz questions and answers
quiz_data = [
    {"question": "What is the most widely used renewable energy source?", "answer": "solar"},
    {"question": "Which renewable energy source is harnessed from the wind?", "answer": "wind"},
    {"question": "Which renewable energy source involves converting water flow into energy?", "answer": "hydropower"},
    {"question": "What is the primary gas contributing to global warming?", "answer": "carbon dioxide"},
    {"question": "Which renewable energy source is generated from organic materials?", "answer": "biomass"},
]

# Train spaCy model with more epochs and improved data
def train_spacy_model():
    # Create a blank English NLP model
    nlp = spacy.blank("en")
    
    # Add text classification pipe to the model
    textcat = nlp.add_pipe("textcat", last=True)
    
    # Add labels to the text classifier
    for _, label in training_data:
        textcat.add_label(label)
    
    # Start training
    optimizer = nlp.begin_training()

    # Training loop for multiple iterations
    for epoch in range(30):  # Increased the number of epochs for better learning
        random.shuffle(training_data)
        losses = {}
        
        for text, label in training_data:
            # Create a doc from the input text
            doc = nlp.make_doc(text)
            # Create an Example object to represent the input/output pair
            example = Example.from_dict(doc, {"cats": {label: 1.0}})
            # Update the model with the example and calculate losses
            nlp.update([example], losses=losses, drop=0.4, sgd=optimizer)

        # Print the loss after every epoch for monitoring the training
        print(f"Epoch {epoch} Losses {losses}")
        
    return nlp

# Load model
nlp_model = train_spacy_model()

# Predict intent with confidence threshold
def predict_intent(text, threshold=0.5):
    doc = nlp_model(text)
    predicted_label, confidence = max(doc.cats.items(), key=lambda item: item[1])

    # Debugging: print the confidence of predictions for debugging
    print(f"Prediction: {predicted_label}, Confidence: {confidence}")
    
    if confidence < threshold:
        return "irrelevant"  # If confidence is low, classify as irrelevant
    return predicted_label

# Get response with fun fact randomly
def get_response(intent, score=None):
    if intent == "irrelevant":
        return "I'm sorry, I didn't quite catch that. Could you rephrase?"
    
    # Select response based on intent
    if intent in responses:
        bot_response = random.choice(responses[intent])
        
        # Add fun fact randomly with a higher chance
        if random.random() < 0.4:  # 40% chance to add a fun fact
            fun_fact = random.choice(fun_facts)
            return f"{bot_response}\n\nFun Fact: {fun_fact}"

        return bot_response
    return "I'm sorry, I didn't quite catch that. Could you rephrase?"

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "Chat History", "About", "Trivia"])

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

    elif app_mode == "Trivia":
        st.title("🔋 Renewable Energy Trivia")
        st.subheader("Let's test your knowledge!")

        if "quiz_question" not in st.session_state:
            st.session_state.quiz_question = random.choice(quiz_data)
            st.session_state.user_answer = ""
        
        question = st.session_state.quiz_question
        correct_answer = question["answer"]

        user_answer = st.text_input(f"Q: {question['question']}")

        if user_answer:
            if user_answer.lower() == correct_answer:
                st.success("Correct! 🎉")
                st.session_state.quiz_question = random.choice(quiz_data)
                st.session_state.user_answer = user_answer
                st.session_state.chat_history.append(("You", user_answer))
                st.session_state.chat_history.append(("Bot", f"Correct! The answer was '{correct_answer}'."))
            else:
                st.error(f"Oops! The correct answer was '{correct_answer}'. Better luck next time!")
                st.session_state.quiz_question = random.choice(quiz_data)
                st.session_state.user_answer = user_answer
                st.session_state.chat_history.append(("You", user_answer))
                st.session_state.chat_history.append(("Bot", f"Incorrect! The correct answer was '{correct_answer}'."))

    elif app_mode == "Chat History":
        st.title("📜 Chat History")
        if "chat_history" in st.session_state and st.session_state.chat_history:
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(f"**{sender}:** {message}")
                else:
                    st.markdown(f"**Bot:** {message}")
        else:
            st.warning("No chat history available.")

    elif app_mode == "About":
        st.title("🌿 About This Bot 🌿")
        st.write("This bot is designed to answer questions related to renewable energy, certifications, career opportunities, and trivia. Feel free to ask about solar, wind, hydropower, and other green energy topics.")
        st.write("It also has a fun trivia section to test your knowledge!")

if __name__ == "__main__":
    main()
