from Vector_db import retrieve_from_qdrant
import logging

logging.basicConfig(
    filename="app_logs.log",  # Log file name
    level=logging.DEBUG,      # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

def retrieve_context(query):
    """ Retrieves query result from Qdrant"""
    results = retrieve_from_qdrant(query)

    # logging.info(f"Retrieved results: {results}")

    context = ''
    if results:
        for hit in results:
            payload = hit.payload
            if 'question' in payload and 'response' in payload:
                context += f"Q:{payload['question']}\nA:{payload['response']}\n"
            elif 'content' in payload:
                context += f"{payload['content']}\n"

    logging.info(f"Generated context  in Prompt file: {context}")    
    # print(f"DEBUG  context: {context}")

    prompt = f"""
    You are an Opkey Assistant, use the below context to form your answer:


    <context>
    {context}
    </context>

    When responding:
    - Always  answer the user's query using the context above.
    - For answering First Scan the context first and than generate respones according to it.
    - If the answer is not found in the context, politely deny the request and begin your response with "Sorry". Clearly inform the user that you can only answer Opkey-related questions.
    - Do not answer general knowledge questions or queries unrelated to the context.
    """
    
    return prompt

