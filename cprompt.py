from dotenv import load_dotenv
import os
import openai
load_dotenv() 
api_key = os.getenv("API_KEY")
classification_prompt= """
    You are an expert in Natural Language Processing (NLP) and Question Classification. Your task is to classify a given question into one or more of the following five categories based on its structure, intent, and required reasoning depth.
    
    1. *Factual*
    A question belongs to the Factual category if:
    - The answer is a single, definitive piece of information
    - It can be directly verified through reliable reference sources
    - No reasoning, inference, or multi-step calculations are required
    - The question explicitly asks "What," "Who," "When," or "Where" in a straightforward manner

    *Examples:*
    - How much did HPE return to shareholders in Q3 2024 through dividends and share repurchases?
    - What was HPE’s operating profit margin for the Hybrid Cloud segment in Q4 2024?
    - When was the dividend for Q4 2024 scheduled to be paid?
    - How much was HPE’s Financial Services segment’s net portfolio assets in Q4 2024?

    2. *Aggregate/Time-Based*
    A question belongs to the Aggregate/Time-Based category if:
    - The answer requires data collection over a defined time period
    - It involves summarization, trend analysis, or numerical aggregation
    - The focus is on changes, comparisons, or averages over time
    - A timeframe is explicitly mentioned or implied

    *Examples:*
    - What was the highest recorded revenue for any quarter in fiscal year 2024?
    - How has HPE’s ARR changed from Q1 2023 to Q4 2024?
    - Compare HPE’s Intelligent Edge operating margins over the last four quarters.
    - Which quarter in fiscal year 2024 had the highest operating profit margin for the Server segment?
    - Track the changes in Hybrid Cloud revenue from Q1 2023 to Q4 2024.

    3. *Global Sensing*
    A question belongs to the Global Sensing category if:
    - It requires understanding large-scale, complex interactions
    - The answer involves multi-perspective contextual analysis
    - It necessitates systemic thinking beyond a single dataset or entity
    - Interdisciplinary knowledge (e.g., finance, business strategy) is needed

    *Examples:*
    - Summarize the revenue trends and executive statements for all quarters in 2024.
    - How did the Juniper Networks acquisition impact HPE’s 2024 outlook?
    - What major financial milestones did HPE achieve in fiscal year 2024?
    - Provide a breakdown of HPE’s profitability in 2024.
    - What were the biggest financial takeaways from HPE’s earnings reports in 2024?

    4. *Reasoning/Inference*
    A question belongs to the Reasoning/Inference category if:
    - It requires extrapolation beyond existing data
    - The answer involves logical deduction, predictive analysis, or causal reasoning
    - It explores potential scenarios, outcomes, or explanations
    - A direct, factual answer is not sufficient

    *Examples:*
    - Based on past trends, which segment is likely to experience the highest revenue growth in 2025?
    - What factors contributed to HPE’s decline in gross margins throughout 2024?
    - How might the Juniper Networks acquisition influence HPE’s networking revenue in 2025?
    - Why did HPE increase its full-year EPS guidance in Q3 2024?

    5. *Multi-Part*
    A question belongs to the Multi-Part category if:
    - It contains multiple interconnected components
    - The response requires analyzing different aspects together
    - A single-step answer is insufficient
    - Synthesis and integration of various perspectives are necessary

    *Examples:*
    - When was HPE's ARR $1.7 billion, and what was the free cash flow that year?
    - What were HPE’s revenue and net earnings per share (EPS) in Q3 2024, and how did they compare to Q2 2024?
    - List the operating profit margins for all business segments in Q4 2024 and their percentage change from the prior year.
    - How did HPE’s Intelligent Edge segment perform in Q1 and Q4 2024, and what factors contributed to the changes?

    *Final Notes:*
    - Don't give the explanation.Only give the relevant classes.
    -Choose the most relevant class from the inputs given.Dont give any other output other than the classes themselves
    - A question can belong to multiple categories if it meets the criteria for more than one classification , return both the categoriy but tell the preference .
    - Prioritize accuracy—misclassification can impact downstream analysis.
    
    Classify the following question accordingly:
"""
def ai_response(ui):
    client = openai.OpenAI(api_key=api_key, base_url="https://models.inference.ai.azure.com")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a question classification expert."},
            {"role": "user", "content": classification_prompt+ ui}
        ]
    )
    
    return response.choices[0].message.content
# Function to get AI response
def get_ai_response(user_input,r1,r2):
    client = openai.OpenAI(api_key=api_key, base_url="https://models.inference.ai.azure.com")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a question classification expert."},
            {"role": "user", "content": classification_prompt+ user_input + "First result: "+ r1 +" second result: "+ ",".join(r2)}
        ]
    )
    
    return response.choices[0].message.content
