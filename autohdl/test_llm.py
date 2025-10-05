from .llm import OpenAI
from .linter import linter
from .data import data, extract_header


def main():
    llm = OpenAI()
    
    ds = data()
    
    message = ds['description'][0]["high_level_global_summary"]
    expected = ds['code'][0]
    
    print("-------------------------------------------")
    print("System prompt:")
    print(llm.messages[0]['content'])
    print("-------------------------------------------\n")
    
    print("-------------------------------------------")
    print("User prompt:\n", message)
    print("-------------------------------------------\n")
    
    print("-------------------------------------------")
    print("Expected:\n", expected)
    print("-------------------------------------------\n")
    
    response = llm(message)
    print("-------------------------------------------")
    print("Response:\n", response)
    print("-------------------------------------------\n")
    
    header = extract_header(message)
    lint_input = header + "\n" + response
    print("-------------------------------------------")
    print("Linter input:\n", lint_input)
    print("-------------------------------------------\n") 
    
    lint_result = linter(lint_input)
    print("-------------------------------------------")
    print("Linter result:\n", lint_result)
    print("-------------------------------------------\n")

    second_response = llm(lint_result)
    print("-------------------------------------------")
    print("Second response:\n", second_response)
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()