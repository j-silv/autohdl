import streamlit as st
from autohdl.data import data
from autohdl.llm import system_prompt, LLM
import random

"""
# AutoHDL
### AI agent which generates Verilog code
"""

def random_sample_btn(stop):
    """Generate a new sample"""
    
    idx = random.randrange(stop)
    st.session_state['idx'] = idx
    print(idx)
    return idx

def generate_btn(model, text):
    st.session_state['response'] = model(text)
    return st.session_state['response']

@st.cache_resource(show_spinner="Loading LLM...")
def load_model():
    model = LLM()
    model.load_model()
    return model 


def server():
    ds = data(small_dataset=True)
    
    model = load_model()
    
    if 'idx' not in st.session_state:
        st.session_state['idx'] = 0
    
    if 'response' not in st.session_state:
        st.session_state['response'] = "Click generate for LLM to respond"
        
    idx = st.session_state['idx']
        
    summary = "high_level_global_summary"
    
    st.text_area("System prompt",
                 system_prompt,
                 height="content")
    
    description_prompt = ds['description'][idx][summary]
    
    st.text_area("User prompt",
                 description_prompt,
                 height=200)
    
    st.text_area("Expected response",
                 ds['code'][idx],
                 height=200)
    
    st.button("Random sample", on_click=random_sample_btn, args=[ds.num_rows])
    
    st.text_area("LLM response",
                 st.session_state['response'],
                 disabled=True,
                 height=200)
    
    st.button("Generate", on_click=generate_btn, args=[model, ds['description'][idx][summary]])
    

if __name__ == "__main__":
    server()