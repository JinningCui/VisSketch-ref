import re


class Parser:
    def parse(self, response):
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        content = str(response).replace(r"\_", "_")
        
        try:
            matches = re.findall(r"```(?:python|py)\s*\n?(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
            if not matches:
                return {
                    'status': False,
                    'content': content,
                    'message': 'No executable Python block found. Use exactly one fenced ```python code block for ACTION, or final ANSWER outside code.',
                    'error_code': 'missing_python_block',
                }
            if len(matches) > 1:
                return {
                    'status': False,
                    'content': content,
                    'message': f'Found {len(matches)} Python code blocks. Use exactly one ACTION code block.',
                    'error_code': 'multiple_python_blocks',
                }

            program = matches[0].strip()
            if not program:
                return {
                    'status': False,
                    'content': program,
                    'message': "The Python code block is empty.",
                    'error_code': 'empty_python_block',
                }
            compile(program, "prog.py", "exec")
            return {'status': True, 'content': program, 'message': 'Parsing succeeded.', 'error_code': ''}
        except Exception as err:
            return {'status': False, 'content': content, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'python_compile_error'}
    
def main():
    parser = Parser()
    
    # testing 1
    program = """Thought: I thought a lot and here is what I am thinking.\nAction:```python\n"""
    program += """def solve():\n"""
    program += """    output0 = text_generation(prompt="Would you rather have an Apple Watch - or a BABY?")\n"""  
    program += """    output1 = text_summarization(text=output0["text"])\n"""
    program += """    return output1\n""" 
    program += """```"""
    print(program)
    results = parser.parse(program)
    print(results)
    
    print("\n\n-----------------------------------\n\n")
    
    # testing 2
    program = """Thought: I thought a lot and here is what I am thinking.\nAction:```python\n"""
    program += """def solve():\n"""
    program += """    aha\noutput0 = text_generation(prompt="Would you rather have an Apple Watch - or a BABY?")\n"""  
    program += """    output1 = text_summarization(text=output0["text"])\n"""
    program += """    return output1\n""" 
    program += """```"""
    print(program)
    results = parser.parse(program)
    print(results)
    
    print("\n\n-----------------------------------\n\n")
    
    # testing 3
    program = """Thought: I thought a lot and here is what I am thinking.\nAction: No need"""
    print(program)
    results = parser.parse(program)
    print(results)
    
    
if __name__ == '__main__':
    main()
