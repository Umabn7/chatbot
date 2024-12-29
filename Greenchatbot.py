from transformers import pipeline
import json
import random
import streamlit as st
intent_classifier = pipeline("text-classification", model="bert-base-uncased")

intents = {
    "course_info": [
        "We offer solar energy, wind energy and waste management courses.",
        "Our courses include solar, wind energy and waste management." ],
    "career_guidance" : [
        "The field of renewable energy is in high demand. Would you like any suggestions for certifactions?!",
        "Green energy careers are booming nowadays and roles like sustainability analyst and energy consultant are popular!"],
    "certificaion_help" : [
        "You can apply for certifications.. Would you like a link??",
        "Green Certifications can boost your career.. Apply and learn more!!"
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
def get_response(user_input):
    prediction = intent_classifier(user_input)[0]
    intent = prediction['label'].lower()
    confidence = prediction['score']

    if intent in intents and confidence > 0.7:
        return random.choice(intents[intent])
    else:
        return "Sorry, I din't understand. Can you rephrase your question"

def main():
    st.title("Green Chat Bot")
    st.subheader("Ask me about courses, career guidance, certifications and more!!")

    user_input = st.text_input("Your message : ")

    if st.button("Send"):
        if user_input.strip():
            response = get_response(user_input)
            st.text_area("Chatbot Response : ", response, height=100)

        else:
            st.warning("Please enter a message to chat!!")

if __name__ == "__main__":
    main()


