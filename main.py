import os
import re
import pdfplumber
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

def read_txt(file_path: str) -> str:
    """Read a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(file_path: str) -> str:
    """Read a PDF file and extract text."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# ============================
# Extract UBR Blocks
# ============================
def extract_ubrs(file_content: str) -> dict:
    """
    Split file content into UBR blocks.
    Each block starts with 'UBR <number>:'
    """
    ubr_blocks = re.split(r"(UBR\s+\d+:)", file_content)
    ubr_dict = {}
    for i in range(1, len(ubr_blocks), 2):
        ubr_id = ubr_blocks[i].strip().replace(":", "")
        ubr_text = ubr_blocks[i+1].strip()
        ubr_dict[ubr_id] = ubr_text
    return ubr_dict


# ============================
# Rate Card Tool
# ============================
@tool
def rate_card_tool(ubr_id: str, ubr_text: str, learners: int, days: int) -> dict:
    """
    Parse UBR instructions and generate a rate card.
    """
    llm = ChatOpenAI(model="gpt-5-mini")
    prompt = f"""
    UBR ID: {ubr_id}
    Instructions: {ubr_text}

    Parameters:
    - Learners: {learners}
    - Days: {days}

    Task:
    Calculate the course fee based on the UBR instructions.
    Return ONLY in this exact format (no extra text):

    UBR {ubr_id}:
    Currency = <value>
    Course_Fee = <value>
    Term = <value>
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ============================
# Build Agent
# ============================
llm = ChatOpenAI(model="gpt-5-mini")
agent = create_agent(llm, tools=[rate_card_tool])

# ============================
# Example Run
# ============================
if __name__ == "__main__":
    # Read file (replace with your actual file path)
    file_content = read_txt(r"D:\Pricing_Project_AI\PRICING_PROJECT_AI\UBR.txt")  # or read_pdf("UBR.pdf")
    ubr_dict = extract_ubrs(file_content)

    # Loop through all UBRs
    outputs = []
    for ubr_id, ubr_text in ubr_dict.items():
        result = agent.invoke({
            "messages": [
                HumanMessage(content=f"Generate rate card for {ubr_id}: {ubr_text}")
            ]
        })
        crisp_output = result["messages"][-1].content
        print(crisp_output)

        outputs.append(crisp_output)

        # print(f"{ubr_id}:\n{result['messages'][-1].content}\n")

    # Write all outputs to a file
    with open(r"D:\Pricing_Project_AI\PRICING_PROJECT_AI\UBR_Complete.txt", "w", encoding="utf-8") as out_file:
        out_file.write("\n\n".join(outputs))

