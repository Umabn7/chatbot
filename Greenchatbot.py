import random
import spacy
from spacy.training import Example
import streamlit as st

# Enhanced training data with more specific examples
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
    ("Hi", "greeting"),
    ("Hello", "greeting"),
    ("Good morning", "greeting"),
    ("Good evening", "greeting"),
    ("How are you?", "greeting"),
    ("What’s up?", "greeting"),
    ("Bye", "goodbye"),
    ("See you later", "goodbye"),
    ("Goodbye", "goodbye"),
    ("Thanks", "thank_you"),
    ("Thank you", "thank_you"),
    ("Thanks for your help", "thank_you"),
    ("Can you help with climate change?", "climate_change"),
    ("What can we do to reduce carbon emissions?", "climate_change"),
    ("Tell me about sustainable practices", "sustainable_practices"),
    ("How can I save energy in my home?", "energy_saving_tips"),
    ("How do solar panels work?", "solar_panels_info"),
    ("How can I reduce my carbon footprint?", "carbon_footprint"),
    ("Tell me about electric vehicles", "electric_vehicles"),
    ("How do wind turbines work?", "wind_turbines_info"),
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
        "Hello! How can I assist you in your green energy journey today?",
        "Hi! I'm here to help you with all things renewable energy and sustainability!",
        "Good day! How can I help you with renewable energy today?"
    ],
    "goodbye": [
        "Goodbye! Stay green and sustainable!",
        "See you later! Keep working towards a greener future!"
    ],
    "thank_you": [
        "You're welcome! I'm always here to help.",
        "Glad I could assist! Feel free to reach out anytime."
    ],
    "climate_change": [
        "To combat climate change, it's crucial to reduce carbon emissions and switch to renewable energy sources.",
        "Fighting climate change involves global efforts like transitioning to clean energy and reducing waste."
    ],
    "sustainable_practices": [
        "Sustainable practices include using renewable energy, reducing waste, and conserving water.",
        "Consider reducing your environmental impact by adopting eco-friendly practices like recycling and supporting green businesses."
    ],
    "energy_saving_tips": [
        "To save energy, consider switching to energy-efficient appliances and using natural light whenever possible.",
        "Unplug devices when not in use, and try to use energy-efficient lighting like LED bulbs."
    ],
    "solar_panels_info": [
        "Solar panels convert sunlight into electricity using photovoltaic cells, providing a clean and renewable energy source.",
        "With solar energy, you can reduce your electricity bills while contributing to a greener planet!"
    ],
    "carbon_footprint": [
        "You can reduce your carbon footprint by driving less, using public transport, and switching to renewable energy sources.",
        "Eating less meat, reducing waste, and using less plastic are other great ways to lower your carbon footprint."
    ],
    "electric_vehicles": [
        "Electric vehicles (EVs) run on electricity and produce zero emissions, making them an excellent choice for reducing your carbon footprint.",
        "The adoption of electric vehicles is one of the key strategies to combat climate change."
    ],
    "wind_turbines_info": [
        "Wind turbines harness the power of wind to generate electricity. They are a key component of renewable energy infrastructure.",
        "As the wind blows, the blades of the turbine spin, which drives a generator that produces electricity."
    ],
}

fun_facts = [
    "Did you know? The energy from the sun in one hour is enough to power the Earth for a year!",
    "Wind turbines can reach heights taller than the Statue of Liberty!",
    "Recycling one aluminum can saves enough energy to power a TV for three hours.",
    "Hydropower is the oldest form of renewable energy, dating back to ancient Greece!",
    "The largest solar farm in the world is located in the Mojave Desert, California.",
    "It takes about 10 years for a wind turbine to offset the carbon emissions it took to manufacture it.",
    "A single solar panel can power a lightbulb for over 10 hours!",
    "The largest wind turbine in the world is over 220 meters tall!"
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
def get_response(intent):
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
    st.title("Renewable Energy Assistant")
    st.write("Welcome! Ask me anything about renewable energy, sustainability, and more.")
    
    # Create columns for the navigation tiles
    col1, col2, col3 = st.columns(3)

    # Create clickable tiles
    with col1:
        if st.button("Home"):
            st.session_state.page = "Home"
    with col2:
        if st.button("Chat History"):
            st.session_state.page = "Chat History"
    with col3:
        if st.button("About"):
            st.session_state.page = "About"

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if st.session_state.page == "Home":
        user_input = st.text_input("Ask a question:")
        if user_input:
            intent = predict_intent(user_input)
            response = get_response(intent)
            st.write(response)

    elif st.session_state.page == "Chat History":
        st.write("Here you can view the chat history!")
        # Add functionality to view chat history

    elif st.session_state.page == "About":
        st.write("This chatbot is designed to help you with information on renewable energy, sustainability, and more. Ask away!")

if __name__ == "__main__":
    main()
