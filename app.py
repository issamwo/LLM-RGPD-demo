import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

from langchain.chains import LLMChain
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    )



## Function to get response from LLAMA 2 model

def get_llama_response(input_text):

    ## llama 2 model : 
    llm = CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens':2000,
                              'temperature':0.01})
    
    
    examples = [
    {"input": "my name is Carlos and my phone number is 0758040882", "output": "my name is NAME and my phone number is PHONE"},
    {"input": "i live at the 23 rue pasteur, you can send me an email at Lily@gmail.com when you visit my city", "output": "i live at the ADRESS, you can send me an email at EMAIL when you visit my city"},
    {"input": "name is Mathieu, and i live the residence soleil", "output": "name is NAME, and i live the ADRESS"},
    {"input": "je m'appelle Lily et mon numéro de téléphone est le 0758040882", "output": "je m'appelle NAME et mon numéro de téléphone est le PHONE"},
    {"input": "j'habite au 50 Boulevard Barcelone, vous pouvez m'envoyer un email à work.millo@gmail.com lorsque vous visitez ma ville", "output": "j'habite au ADRESS, vous pouvez m'envoyer un email à EMAIL lorsque vous visitez ma ville"},
    {"input": "je m'appelle Ahmed et j'habite à la résidence Trinu", "output": "je m'appelle NAME et j'habite à la résidence ADRESS"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    )


    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are NER: A Named Entity Recognition (NER) system identifies and classifies named entities\
              such as people, organizations, locations, and other types of proper nouns in a text. that reads the text and \
              masks all personnal informations such as name, adress, email, phone number ... you don't change anything in the the original text or translate it, the output shoud be the anonymized input and nothing else"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    
 
    chain = LLMChain(
        llm=llm,
        prompt=final_prompt,
        verbose=True
    )
    
    print(chain.run(input_text)[0])

    return chain.run(input_text)



st.set_page_config(page_title="RGPD Officer",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("RGPD Officer 🤖")

input_text=st.text_input("Enter text to be ANONYMIZED (mask personnal info)")


submit = st.button('Anonymize text')

## Final response:

if submit:
    st.write(get_llama_response(input_text))