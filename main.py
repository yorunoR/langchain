import os
from dotenv import load_dotenv
from langchain.memory import ConversationKGMemory
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.should be output japanese.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)
memory=ConversationKGMemory(llm=llm)
conversation_with_kg = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=prompt,
    memory=memory
)

print('1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(conversation_with_kg.predict(input="僕の名前は茜です。"))
print('\n\n\n\n\n\n\n\n2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(conversation_with_kg.predict(input="僕の名前は？"))
print('\n\n\n\n\n\n\n\n3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(conversation_with_kg.predict(input="今日は雨でした。"))
print('\n\n\n\n\n\n\n\n4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(conversation_with_kg.predict(input="今日の天気は？"))
