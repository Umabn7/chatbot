import random
import streamlit as st

# Define intents and responses
intents = {
    "course_info": [
        "We offer solar energy, wind energy and waste management courses.",
        "Our courses include solar, wind energy and waste management."
    ],
    "career_guidance": [
        "The field of renewable energy is in high demand. Would you like any suggestions for certifications?!",
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

# Keyword-based intent matcher
keywords = {
    "course_info": ["course", "training", "program"],
    "career_guidance": ["career", "job", "guidance"],
    "certification_help": ["certification", "certificate", "certified"],
    "renewable_energy_advantages": ["advantages", "benefits", "pros"],
    "renewable_energy_challenges": ["challenges", "problems", "issues"],
    "environmental_tips": ["tips", "environment", "save"],
    "renewable_energy_trends": ["trends", "latest", "update"],
    "job_opportunities": ["job", "roles", "opportunity"]
}

# Function to get chatbot response
def get_response(user_input):
    user_input = user_input.lower()
    for intent, words in keywords.items():
        if any(word in user_input for word in words):
            return random.choice(intents[intent])
    return "Sorry, I didn't understand. Can you rephrase your question?"

# Streamlit App
def main():
    st.title("Green Chat Bot")
    st.subheader("Ask me about courses, career guidance, certifications, and more!")

    user_input = st.text_input("Your message:")

    if st.button("Send"):
        if user_input.strip():
            response = get_response(user_input)
            st.text_area("Chatbot Response:", response, height=100)
        else:
            st.warning("Please enter a message to chat!")

if __name__ == "__main__":
    main()
