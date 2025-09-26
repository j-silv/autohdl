import streamlit as st
from autohdl.data import data
from autohdl.llm import system_prompt, LLM
import random
from code_editor import code_editor

"""
# AutoHDL
### AI agent which generates Verilog code
"""
def code_input(title, message, edit_disabled=False):
    st.text(title)
    
    ace_props = {"style": {"borderRadius": "0px 0px 8px 8px"}}
    
    code_editor(message,
                height = 10,
                lang="text",
                theme="default",
                shortcuts="vscode",
                focus=False,
                props=ace_props,
                response_mode="debounce",
                options={"wrap": True})
    

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
    
    row1 = st.container(border=True, height=400)
    row2 = st.container(border=True, height=400)
    
    with row1:
        col1, col2 = st.columns(2)
        
        with col1:
            code_input("System prompt", system_prompt)
        
        with col2:
            description_prompt = ds['description'][idx][summary]
            code_input("User prompt", description_prompt)   
            
    with row2:
        col1, col2 = st.columns(2)
        
        with col1:
            code_input("Expected response", ds['code'][idx])

    
        with col2:
            code_input("LLM response", st.session_state['response'], edit_disabled=True)
    
    with st.container(horizontal=True):
        st.button("Random sample", on_click=random_sample_btn, args=[ds.num_rows])
        st.button("Generate", on_click=generate_btn, args=[model, description_prompt])
    

if __name__ == "__main__":
    server()