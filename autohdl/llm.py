import outlines
from outlines.inputs import Chat
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import data

system_prompt = ("You only complete chats with syntax correct Verilog code. "
                 "End the Verilog module code completion with 'endmodule'. "
                 "Do not include module, input and output definitions.")

class LLM:
    def __init__(self):
        self.system_prompt = system_prompt
        
    def load_model(self, use_cpu=True):
        if use_cpu:
            model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
            self.device = "cpu"
        else:
            model_name = "codellama/CodeLlama-7b-Instruct-hf"
            self.device = "cuda"

        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        return self.hf_tokenizer, self.hf_model

    def __call__(self, description_prompt):

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": description_prompt}
        ]

        input_text=self.hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(input_text)
        
        inputs = self.hf_tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        outputs = self.hf_model.generate(inputs)
        
        # print(self.hf_tokenizer.decode(outputs[0]))
        return self.hf_tokenizer.decode(outputs[0])

