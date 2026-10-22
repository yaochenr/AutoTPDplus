import re
import json

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field

from os import getenv
from dotenv import load_dotenv


response_schemas = [
    ResponseSchema(name="Compound_Name", description="Name of the compound", type="string"),
    ResponseSchema(name="IUPAC_Name", description="IUPAC name of the compound. (optional)", type="string"),
    ResponseSchema(name="SMILES", description="SMILES string of the compound. (optional)", type="string"),
    ResponseSchema(name="Degradation_Target", description="The target protein to be degraded", type="string"),
    ResponseSchema(name="Recruiter", description="The E3 ligase (or E3 ligase-recruiting protein) involved in the degradation process.", type="string"),
    ResponseSchema(name="Assay", description="The assay method used to measure degradation", type="string"),
    ResponseSchema(name="Cell_Line", description="The specific cell line used for the assay; if a cell-free assay is used, denote this as 'cell-free'", type="string"),
    ResponseSchema(name = "DC50", description = "The compound concentration required to achieve 50% degradation of the target protein. (optional)"),
    ResponseSchema(name = "DC50_units", description = "The units of DC50 measurement.(optional)"),
    ResponseSchema(name = "DC50_h", description = "Time point (in hours) at which the DC50 measurement was obtained. (optional)"),
    ResponseSchema(name = "Dmax", description="The maximal degradation observed (out of 100%). (optional)"),
    ResponseSchema(name = "Dmax_h", description="Time point (in hours) at which the Dmax measurement was obtained. (optional)"),
    ResponseSchema(name = "Dmax_concentration", description="The compound concentration at which the maximum degradation (Dmax) was observed (e.g., 100 nM, 1 μM). (optional)")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions(only_json=True)

def run_llm_api(custom_prompt=None, paper_text=None, temperature=0):
    load_dotenv()  

    if paper_text is not None:
        custom_template = """{custom_prompt}
        {format_instructions}
        Please output only valid JSON, without any additional commentary or explanations.
        Please note: even if there is only one record, output it as a JSON array (for example: [{{…}}]).
        For this paper:
        {paper_text}
        Please extract the data according to the above guidelines and format instructions."""

        prompt_input = PromptTemplate(template=custom_template, input_variables=["custom_prompt", "format_instructions", "paper_text"])
        prompt_kwargs = {
            "custom_prompt": custom_prompt,
            "format_instructions": format_instructions,
            "paper_text": paper_text}
        print(f"Input to model: {prompt_kwargs}")
        print("=" * 120)
    
    else:
        formatted_prompt = custom_prompt or ""
        print("Input to model:", formatted_prompt)


    llm = ChatOpenAI(
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base=getenv("OPENROUTER_BASE_URL"),
        model_name=getenv("LLM_MODEL_NAME"),
        temperature=temperature,
        request_timeout=1200,
    )

    if paper_text is not None:
        chain = prompt_input | llm
        response = chain.invoke(prompt_kwargs)
        print(f"Original response from model: {response}")
        print(f"Type of original response from model: {type(response)}")
        
        response_content = response.content
        response_clean = re.sub(r"^```json|```$", "", response_content, flags=re.MULTILINE).strip()
        
        try:
            parsed_response = json.loads(response_clean)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed in run_llm_api: {e}")
            print(f"Raw response content (first 500 chars): {response_content[:500]}")
            print(f"Cleaned response content (first 500 chars): {response_clean[:500]}")
            return response_content, response.response_metadata.get('model_name', 'unknown')

        if isinstance(parsed_response, dict):
            records = [parsed_response]
        elif isinstance(parsed_response, list):
            records = parsed_response
        else:
            raise ValueError(f"Unexpected JSON structure: {type(parsed_response)}")
        
        model_name = response.response_metadata.get('model_name', 'unknown')

    else:
        records = llm.predict(formatted_prompt)
        print(f"Original response from model: {records}")
        model_name = llm.model_name

    return records, model_name






def debug_print_prompt(prompt, prompt_kwargs, history_messages):

    try:
        rendered_prompt = prompt.format_messages(
            **prompt_kwargs,
            history=history_messages
        )
        print("=" * 120)
        print("🔍 Final prompt that will be sent to the model:")
        for msg in rendered_prompt:
            role = msg.type.capitalize()
            print(f"\n[{role}]:\n{msg.content}")
        print("=" * 120)
    except Exception as e:
        print("⚠️ Failed to render prompt for debug:", e)


def run_llm_api_with_history(session_id: str, custom_prompt: str, paper_text: str | None = None, temperature=0, get_session_history=None):
    load_dotenv()

    if get_session_history is None:
        raise ValueError("get_session_history function must be provided")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in chemistry."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base=getenv("OPENROUTER_BASE_URL"),
        model_name=getenv("LLM_MODEL_NAME"),
        temperature=temperature,
        request_timeout=1200,
    )

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    question_txt = custom_prompt if paper_text is None else custom_prompt
    response = chain_with_history.invoke(
        {"question": question_txt},
        config={"configurable": {"session_id": session_id}},
    )

    raw = response.content

    print(f"[DEBUG] Raw response length: {len(raw) if raw else 0}")
    print(f"[DEBUG] Raw response (first 500 chars): {raw[:500] if raw else 'EMPTY'}")
    print(f"[DEBUG] Response metadata: {response.response_metadata}")

    cleaned = re.sub(r"^```json|```$", "", raw, flags=re.I | re.M).strip()
    try:
        if cleaned and cleaned[0] in "{[":
            parsed_response = json.loads(cleaned)
            if isinstance(parsed_response, dict):
                records = [parsed_response]
            elif isinstance(parsed_response, list):
                records = parsed_response
            else:
                raise ValueError(f"Unexpected JSON structure: {type(parsed_response)}")
            
            model_name = response.response_metadata.get('model_name', 'unknown')

            return records, model_name
        else:
            model_name = response.response_metadata.get('model_name', 'unknown')
            return raw, model_name
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        print(f"Raw response content (first 500 chars): {raw[:500]}")
        print(f"Cleaned response content (first 500 chars): {cleaned[:500]}")
        return raw, response.response_metadata.get("model_name", "unknown")


def construct_prompt(custom_prompt: str, format_instructions: str = format_instructions, paper_text: str | None = None) -> str:

    if not paper_text:
        return custom_prompt.strip()

    tmpl = """{custom_prompt}
        {format_instructions}
        Here is the paper:
        ```
        {paper_text}
        ```
        Please extract the data according to the above guidelines and format instructions."""
    first_round_prompt = tmpl.format(
        custom_prompt=custom_prompt.strip(),
        format_instructions=format_instructions.strip(),
        paper_text=paper_text.strip(),
    )
    return first_round_prompt
