import random
import spacy
import json
from spacy.training import Example
import streamlit as st

# Load intents and responses from intents.json
with open('intents.json') as file:
    data = json.load(file)

# Convert loaded intents into a more usable format
intents = data['intents']

# Create a list of training examples
training_data = []
responses = {}

for intent in intents:
    for example in intent['examples']:
        training_data.append((example, intent['intent']))
    
    # Store responses for each intent
    responses[intent['intent']] = intent['responses']

# Function to train the spaCy model
def train_spacy_model():
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat", last=True)
    
    # Add labels from the intents file
    for intent in intents:
        textcat.add_label(intent['intent'])
    
    optimizer = nlp.begin_training()
    
    for epoch in range(30):  # More epochs for better learning
        random.shuffle(training_data)
        losses = {}
        for text, label in training_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, {"cats": {label: 1.0}})
            nlp.update([example], losses=losses, drop=0.4, sgd=optimizer)
        print(f"Epoch {epoch} Losses {losses}")
    
    return nlp

# Load the trained model
nlp_model = train_spacy_model()

# Function to predict intent with confidence threshold
def predict_intent(text, threshold=0.5):
    doc = nlp_model(text)
    predicted_label, confidence = max(doc.cats.items(), key=lambda item: item[1])
    
    # Print for debugging purposes
    print(f"Prediction: {predicted_label}, Confidence: {confidence}")
    
    if confidence < threshold:
        return "irrelevant"  # If confidence is low, classify as irrelevant
    return predicted_label

# Get a response based on the intent
def get_response(intent):
    if intent == "irrelevant":
        return "I'm sorry, I didn't quite catch that. Could you rephrase?"
    
    # Select a response from the predefined responses
    if intent in responses:
        bot_response = random.choice(responses[intent])
        
        # Fun facts logic - Add 50% chance to return a fun fact
        if random.random() < 0.5:  # 50% chance
            fun_fact = random.choice(responses["fun_facts"])
            return f"{bot_response}\n\nFun Fact: {fun_fact}"

        return bot_response
    return "I'm sorry, I didn't quite catch that. Could you rephrase?"

# Streamlit app for displaying the chat interface
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
