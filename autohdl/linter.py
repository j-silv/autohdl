import subprocess

def linter(code:str)-> str:
    """A tool that runs Verilog syntax linting (with Verilator) on input code string
    Args:
        code: A string representing the Verilog code to be linted
    """
    
    with open("code.v", "w") as f:
        f.write(code)
    
    try:
        subprocess.run(["verilator", "--lint-only", "-Wall", "code.v"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                check=True,
                                encoding="utf-8")
    
        return "The linting passed successfully - no changes necessary"
    
    except subprocess.CalledProcessError as e:
        return f"Verilator linting gave an error. Please investigate and fix: {e.stdout}"
        