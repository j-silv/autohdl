import re
from datasets import load_dataset


def extract_header(prompt):
    """Use regex to extract module header from prompt"""
    
    module_re = re.compile(r"module.*;")

    try: 
        result = re.search(module_re, prompt).group(0)
    except:
        raise Exception("Prompt is not in expected format when extracting header")
    
    return result.strip()

def extract_description(prompt):
    """Use regex to extract description from prompt
    
    The MG-verilog dataset comes with the special tokens such as 
    [INST], <<SYS>>, etc. This removes those and only extracts
    the description and verilog module header. We do this 
    so that we can use models with different chat templates.
    """
    
    sysend_re = re.compile(r"<<\/SYS>>", re.MULTILINE)
    instend_re = re.compile(r"\[\/INST\]", re.MULTILINE)

    try: 
        sysend_pos = re.search(sysend_re, prompt).end()
        instend_pos = re.search(instend_re, prompt).start()
    except:
        raise Exception("Prompt is not in expected format when extracting description")
    
    return prompt[sysend_pos:instend_pos].strip()

def replace_template(batch):
    """Remove system prompt and add raw description text to all summaries"""
    
    for batch_idx in range(len(batch['description'])):
        descriptions = batch['description'][batch_idx]
        
        for summary_type in descriptions:
            descriptions[summary_type] = extract_description(descriptions[summary_type])
    return batch
    

def data(name="GaTech-EIC/MG-Verilog", batch_size=4, small_dataset=True):
    """Load MG-Verilog dataset from GAtech paper
    
    https://arxiv.org/pdf/2407.01910
    """
    
    ds = load_dataset(name, split=f"train[:{10 if small_dataset else ''}]")
    ds = ds.map(replace_template, batched=True, batch_size=batch_size)
    
    return ds
    

if __name__ == "__main__":
    data()
    