import outlines
from outlines.inputs import Chat
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import tiktoken
from dotenv import load_dotenv


class LLM:
    system_prompt = ("You only complete chats with syntax correct Verilog code. "
                    "End the Verilog module code completion with 'endmodule'. "
                    "Do not include module, input and output definitions.")
    
    def __init__(self):
        load_dotenv()


class OpenAI(LLM):
    def __init__(self, system_prompt=None, max_context_len=1000, model="gpt-5-nano"):
        super().__init__()
        
        if system_prompt:
            self.system_prompt = system_prompt
            
        self.max_context_len = max_context_len
        self.model = model
        
        self.messages = [{"role": "system", "content": self.system_prompt}]
        
        self.client = openai.OpenAI()
        
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        
        response = self.client.responses.create(
            model=self.model,
            input=self.messages,
            service_tier="flex"
        )
        
        self.messages.append({"role": "assistant", "content": response.output_text})
        
        return response.output_text

    def truncate(self):
        """Truncate off tokens until we reach max_context_len
        
        Approach is to first determine the index where we are below the max token length
        in the messages array. Once we find this index, then we truncate off that content until
        we are below the max.
        """
        total_len = sum(len(message['content']) for message in self.messages)
        
        i = len(self.messages)-1
        while total_len > self.max_context_len:
            pass
        

class HuggingFace(LLM):
        
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
        
        inputs = self.hf_tokenizer(input_text, return_tensors="pt").to(self.device)
        
        outputs = self.hf_model.generate(**inputs, max_new_tokens=50)
        
        return self.hf_tokenizer.decode(outputs[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

