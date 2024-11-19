#!/usr/bin/env python
# coding: utf-8

import getpass
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

# Set up the API key
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Instantiate the ChatGroq model
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.3,
    max_tokens=500,
    max_retries=3,
    timeout=60,
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are Violet, an advanced AI assistant designed to help developers working with Generative AI. Your tasks include:

            1. Prompt Refinement and Generation:
               - Analyze and optimize user-provided prompts to improve clarity, specificity, and effectiveness.
               - Suggest alternative prompts that yield better results for different models and tasks.

            2. AI Debugging Assistance:
               - Identify errors or inconsistencies in AI-generated outputs or training data.
               - Provide actionable recommendations to fix these issues and improve the overall performance of the Generative AI workflow.

            3. Model Evaluation and Benchmarking:
               - Compare and benchmark the performance of different Generative AI models on similar tasks.
               - Offer detailed insights, such as strengths, weaknesses, and ideal use cases for each model.

            4. Workflow Automation:
               - Assist in integrating Generative AI seamlessly into development pipelines by generating code snippets, workflows, or configurations.
               - Automate repetitive tasks and streamline the development process.

            Your Goals:
            - Provide accurate, actionable, and developer-friendly outputs.
            - Be responsive to user inputs and adapt your recommendations to specific needs or constraints.
            - Ensure that all suggestions are efficient, explainable, and aligned with the best practices in Generative AI.
            """,
        ),
        (
            "human",
            """
            Please provide a structured response to the following task description without explicitly labeling the sections as "Introduction," "Main Points," or "Conclusion." Instead, provide a natural flow where the information is organized logically:

            Task: {task_description}
            """,
        ),
    ]
)

chain = prompt | llm

# Define a function to get responses with dynamic conversation history
conversation_history = ""

def get_response(prompt):
    global conversation_history
    
    # Append the user prompt to the conversation history
    conversation_history += f"User: {prompt}\n"
    
    # Get the model's response
    response = chain.invoke(conversation_history)
    
    # Append the model's response to the conversation history
    conversation_history += f"Assistant: {response.content}\n"
    
    return response.content

if __name__ == "__main__":
    # Example usage
    response1 = get_response("2+2")
    print(response1)