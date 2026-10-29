from langchain.tools import tool

@tool
def tool1():
    """tool1
    """
    
    print("this is tool 1")
   
@tool    
def tool2():
    """tool2
    """
    print("this is tool 2")
    
    
tools = [tool1, tool2]

tool_dict = {t.name: t for t in tools}

print(tool_dict)