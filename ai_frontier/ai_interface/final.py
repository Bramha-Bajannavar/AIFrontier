#!/usr/bin/env python
# coding: utf-8

# # DevHub

# ## API Call

# In[1]:


import getpass
import os
from langchain.prompts import ChatPromptTemplate



if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")


# ## Instantiation

# In[2]:


from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",        # Using LLaMA 3.0 8B model
    temperature=0.3,                # Lower temperature for deterministic, focused output
    max_tokens=500,                 # Moderate token limit for concise, detailed answers
    max_retries=3,                  # Retries up to 3 times in case of failure
    timeout=60,                     # 60 seconds to allow for complex tasks        
)


# ## Llamma 3.0 Testing

# In[3]:




# In[4]:





# ## Training

# In[3]:


from langchain.prompts import ChatPromptTemplate  # Correct import statement

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
            - Remember the previous task description and context for future reference in your responses.
            -Make Sure there is no explicit context
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

chain=prompt | llm


# In[42]:





# In[43]:





# In[4]:


from langchain_core.prompts import PromptTemplate

examples = [
    {
        "prompt": "What is machine learning? Answer concisely in one sentence.",
        "response": "Machine learning is a subset of AI that enables systems to learn from data."
    },
    {
        "prompt": "Explain recursion in programming in simple terms.",
        "response": "Recursion is a function calling itself to solve a problem."
    },
    {
        "prompt": "Write a prompt for a language model to generate a short story about a character named Alex. Can this prompt be any better?",
        "response": "The following Prompt can be written as...Please generate a short story about a character named Alex who is a Software Engineer."
    },
    {
        "prompt": "What is the difference between an API and a UI? Explain in simple terms.",
        "response": "An API is a set of rules that enables different applications to communicate with each other, while a UI is the visual part of a software application."
    },
    {
        "prompt": "Write a script for a conversation between a user and a virtual assistant.",
        "response": "Here is a script for a conversation between a user and a virtual assistant... User: What is the weather like today? Virtual Assistant: The weather is sunny and 75 degrees Fahrenheit."
    },
    {
        "prompt": "Explain the concept of DevOps in the context of software development.",
        "response": "DevOps is the practice of combining software development and IT operations to improve collaboration, efficiency, and quality."
    },
    {
        "prompt": "Write a prompt for a language model to generate a poem about a sunset.",
        "response": "Here is a prompt for a language model to generate a poem about a sunset... Please generate a poem about a sunset with vivid descriptions of colors and emotions."
    },
    {
        "prompt": "Explain the difference between a debugger and a profiler in software development.",
        "response": "A debugger is a software tool that helps developers find and fix errors in their code, while a profiler is a tool that helps developers understand how their code is performing and identify bottlenecks."
    },
    {
        "prompt": "Write a script for a chatbot to order food at a restaurant.",
        "response": "Here is a script for a chatbot to order food at a restaurant... Chatbot: Hi, I'd like to order food. What options do you have? User: We have a variety of dishes. What can I get for you?"
    },
    {
        "prompt": "Explain the concept of cloud computing in simple terms.",
        "response": "Cloud computing is a way of accessing and using computing resources over the internet, instead of having to use and maintain your own hardware and software."
    },
    {
        "prompt": "Write a prompt for a language model to generate a short story about a character who discovers a hidden talent.",
        "response": "Here is a prompt for a language model to generate a short story about a character who discovers a hidden talent... Please generate a short story about a character who discovers a talent for playing the guitar."
    },
    {
        "prompt": "Explain the importance of testing in software development.",
        "response": "Testing is an essential part of software development, as it ensures that the software works correctly and meets the requirements and expectations of the users."
    },
    {
        "prompt": "Write a script for a chatbot to make a reservation at a hotel.",
        "response": "Here is a script for a chatbot to make a reservation at a hotel... Chatbot: Hi, I'd like to make a reservation at your hotel. What dates are available?"
    },
    {
        "prompt": "The model is generating repeated sentences. What could be wrong?",
        "response": "The issue might be caused by insufficient training data variety or low temperature settings during inference. Try increasing the temperature parameter to introduce more randomness in outputs or fine-tune the model with a more diverse dataset."
    },
    {
        "prompt": "How does GPT-3 compare to LLaMA-2 for creative writing tasks?",
        "response": "GPT-3 tends to excel in generating creative, context-rich stories due to its extensive training dataset. LLaMA-2, while efficient, may require fine-tuning for niche creative tasks. If cost and deployment flexibility are priorities, LLaMA-2 is a strong contender."
    },
    {
        "prompt": "How can I integrate a Generative AI model into my Python-based application?",
        "response": "To integrate a Generative AI model into your Python application, you can use libraries like Hugging Face Transformers. First, install the library using `pip install transformers`. Then, load a pre-trained model with `from transformers import AutoModel`. Create a simple API wrapper for your application to send prompts and receive outputs seamlessly."
    },
]


example_prompt = PromptTemplate.from_template("""
User: {prompt}
Assistant: {response}
""")

print(example_prompt.invoke(examples[7]).to_string())

#for example in examples:
    #prompt = example_prompt.format(prompt=example['prompt'], response=example['response'])
    #print(prompt)


# In[5]:


response4=chain.invoke('''what are API's''')
print(response4.content)


# In[6]:


from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("prompt: {prompt}\n response: {response}")

filled_prompt = example_prompt.format(
    prompt="How does OpenAI’s GPT-4 compare to LLaMA-3 for code generation tasks?",
    response='''Here’s a comparative analysis of GPT-4 and LLaMA-3 for code generation:

Performance: GPT-4 demonstrates stronger contextual understanding, making it better for complex code tasks and multi-step logic. LLaMA-3 performs efficiently but may require fine-tuning for niche or domain-specific code tasks.
Cost and Accessibility: LLaMA-3, being open-source, is cost-effective for local deployment. GPT-4 is accessible via APIs but comes with usage fees.
Scalability: LLaMA-3 is well-suited for integration into custom workflows due to its flexibility, while GPT-4 excels in pre-trained tasks without much customization.
Recommendation: Use GPT-4 for immediate, high-quality code generation. Opt for LLaMA-3 if customization or cost-effectiveness is a priority.'''
)

response4=chain.invoke(filled_prompt)
print(response4.content)


# In[15]:


conversation_history = ""

def get_response(prompt):
    global conversation_history
    
    # Append the user prompt to the conversation history
    conversation_history += f"User: {prompt}\n"
    
    # Get the model's response
    response = chain.invoke(conversation_history)
    
    # Append the model's response to the conversation history
    conversation_history += f"Assistant: {response.content}\n"
    
    # Optionally, you can keep updating the chain by appending further prompts
    return response.content


# In[16]:


get_response("what are API's")


# In[17]:


get_response("List Some")


# In[ ]:


if __name__ == "__main__":
    response1 = get_response("2+2")
    print(response1)


