from Vector_db import retrieve_from_qdrant

def retrieve_context(query):
    """ Retrieves query result from Qdrant"""
    results = retrieve_from_qdrant(query)
    print("DEBUG: Retrieved results:", results)

    context = ''
    if results:
        for hit in results:
            payload = hit.payload
            if 'question' in payload and 'response' in payload:
                context += f"Q:{payload['question']}\nA:{payload['response']}\n"
            elif 'content' in payload:
                context += f"{payload['content']}\n"
        
    # print(f"DEBUG  context: {context}")

    prompt = f"""
    You are an Opkey Assistant.

    Context:
    <context>
    {context}
    </context>

    When responding:
    - Always prioritize answering the user's query using the context above.
    - If the answer is not found in the context, politely deny the request and begin your response with "Sorry". Clearly inform the user that you can only answer Opkey-related questions.
    - Do not answer general knowledge questions or queries unrelated to the context.
    """
    
    return prompt

