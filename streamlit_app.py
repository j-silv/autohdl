import streamlit as st
from autohdl.data import data
from autohdl.llm import system_prompt, LLM
import random

"""
# AutoHDL
### AI agent which generates Verilog code
"""
def text_cell(title, message, edit_disabled=False, height="content"):
    st.text(title)
    st.text_area(title,
                 message.strip(),
                 label_visibility="collapsed",
                 height=height,
                 disabled=edit_disabled)    
    
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
    description_prompt = ds['description'][idx][summary]
    
    CELL_HEIGHT = 300
    
    # Prompt cells
    with st.container(border=True, height=CELL_HEIGHT+100):
        left, right = st.columns(2)
        
        with left:
            text_cell("System prompt", system_prompt, height=CELL_HEIGHT)
        
        with right:
            text_cell("User prompt", description_prompt, height=CELL_HEIGHT)  
    
    # Response cells
    with st.container(border=True, height=CELL_HEIGHT+100):
        
        left, right = st.columns(2)
        
        with left:
            text_cell("Expected response", ds['code'][idx], height=CELL_HEIGHT)
        
        with right:
            text_cell("LLM response", st.session_state['response'], edit_disabled=True, height=CELL_HEIGHT)
    
    
    with st.container(horizontal=True):
        st.button("Random sample", on_click=random_sample_btn, args=[ds.num_rows])
        st.button("Generate", on_click=generate_btn, args=[model, description_prompt])
    

if __name__ == "__main__":
    server()