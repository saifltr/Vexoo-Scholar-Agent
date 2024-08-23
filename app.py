import streamlit as st
# from serpapi import GoogleSearch
import serpapi
from mistralai_azure import MistralAzure
from datetime import datetime
import os
import anthropic
import json

st.set_page_config(page_title="Vexoo Scholar", layout="wide")

st.write(
    os.environ["AZURE_AI_ENDPOINT"] == st.secrets["AZURE_AI_ENDPOINT"],
    os.environ["AZURE_AI_API_KEY"] == st.secrets["AZURE_AI_API_KEY"],
    os.environ["serpapi_api_key"] == st.secrets["serpapi_api_key"],
    os.environ["ANTHROPIC_API_KEY"] == st.secrets["ANTHROPIC_API_KEY"]
)

class ScholarSearchEngine:
    
    def fetch_google_scholar_results(self, query, num_results):
        params = {
            "api_key": os.getenv('SERPAPI_API_KEY'),
            "engine": "google_scholar",
            "q": query,
            "num": num_results,
            "sort": "date",
            "as_ylo": datetime.now().year - 5
        }
        
        search = serpapi.search(params)
        results = search.get_dict()
        return results.get("organic_results", [])

    def format_scholar_results(self, search_data):
        formatted_results = []
        for result in search_data:
            formatted_result = {
                'source': result.get('publication_info', {}).get('summary', ''),
                'date': result.get('publication_info', {}).get('summary', ''),
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'link': result.get('link', ''),
                'citations': result.get('inline_links', {}).get('cited_by', {}).get('total', 0)
            }
            formatted_results.append(formatted_result)
        return formatted_results

def generate_research_areas(query):
    client = anthropic.Anthropic()

    system_prompt = f"""
        You are a search engine specialist with expertise in crafting precise, search engine-friendly queries to retrieve the most relevant and insightful information on any given topic. 
        Based on the user query, generate a set of 3 research-focused search queries. Each query should be optimized for search engines, covering key aspects such as historical context, current developments, expert opinions, and diverse perspectives. 
        Ensure that the queries are specific, actionable, and designed to yield high-quality search results.

        Following are the examples:
        Latest news and articles on the current status of topic.
        Expert opinions and analyses on the implications of topic.
        Impact of topic on global/regional politics, economy, and society

        Generate such different queries based on the user query analyse it and craft a good query that can fetch fine results from web
        Output format:

        Strictly output in this format:
        [
        "Research area query 1",
        "Research area query 2",
        "Research area query 3"  
        ]
        """

    user_message = f"User query: {query}"

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=500,
        system=system_prompt,
        messages=[
                {"role": "user", "content": user_message}
        ]
    )
    if message:
        research_queries = json.loads(message.content[0].text)
        return research_queries


def generate_queries(query):
    endpoint = os.environ.get("AZURE_AI_ENDPOINT")
    api_key = os.environ.get("AZURE_AI_API_KEY")

    client = MistralAzure(azure_endpoint=endpoint,
                 azure_api_key=api_key)
    
    generator_prompt = f"""
    Given the set of 3 queries: {query}, generate a set of related questions that dive deeper into the topic, aiming to uncover various aspects, perspectives, or details. 
    The generated questions should be relevant, clear, and designed to prompt further exploration or clarification on the topic. 
    Provide exactly 9 related questions, 3 for each of the given queries.

    Format your response as a simple list of questions, one per line, without numbering or any other formatting.
    """
    
    messages = [
        {"role":"system", "content":generator_prompt},
        {"role":"user", "content":query}
    ]

    resp = client.chat.complete(messages=messages, model="azureai")

    if resp and resp.choices:
        content = resp.choices[0].message.content
        # Split the content by lines and clean up
        questions = [line.strip() for line in content.split('\n') if line.strip()]
        # Ensure we have exactly 9 questions
        return questions[:9] if len(questions) >= 9 else questions + [""] * (9 - len(questions))
    return []
    

def mistral_scholar(query, web_results):
    endpoint = os.environ.get("AZURE_AI_ENDPOINT")
    api_key = os.environ.get("AZURE_AI_API_KEY")

    client = MistralAzure(azure_endpoint=endpoint,
                 azure_api_key=api_key)
    
    prompt = f"""
    You are a knowledgeable research scholar tasked with providing a comprehensive and accurate answer based on the following user query and web search results. 
    Your response should be well-structured, informative, and cite relevant sources when necessary. 
    Consider the credibility, relevance, and recency of the information available in the search results.

    User query: {query}
    Web results from the internet: {web_results}

    Instructions:
    Analyze the Query: Understand the user's question or inquiry.
    Evaluate the Sources: Review the provided web search results, noting the relevance, credibility, and citation count of each source.
    Synthesize Information: Combine insights from the most credible and relevant sources to form a comprehensive and detailed answer.
    Cite Sources: Where necessary, cite the sources in your response to support your statements.

    Output:
    Provide a clear, concise, and well-reasoned answer to the user query based on the information from the web search results. 
    Ensure the answer is suitable for someone seeking a scholarly explanation.
    """

    messages = [
        {"role":"system", "content":prompt},
        {"role":"user", "content":query}
    ]

    resp = client.chat.complete(messages=messages, model="azureai")

    if resp:
        return resp.choices[0].message.content
    
def claude_scholar(query, web_results):
    client = anthropic.Anthropic()

    system_prompt = f"""
    You are a knowledgeable research scholar tasked with providing a comprehensive and accurate answer based on the following user query and web search results. 
    Your response should be well-structured, informative, and cite relevant sources when necessary. 
    Consider the credibility, relevance, and recency of the information available in the search results.

    Instructions:
    Analyze the Query: Understand the user's question or inquiry.
    Evaluate the Sources: Review the provided web search results, noting the relevance, credibility, and citation count of each source.
    Synthesize Information: Combine insights from the most credible and relevant sources to form a comprehensive and detailed answer.
    Cite Sources: Where necessary, cite the sources in your response to support your statements.

    Output:
    Provide a clear, concise, and well-reasoned answer to the user query based on the information from the web search results. 
    Ensure the answer is suitable for someone seeking a scholarly explanation.
    """

    user_message = f"User query: {query}\nWeb results from the internet: {web_results}"

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    return message.content[0].text
def main():    
    st.title("Vexoo Scholar")
    
    # Sidebar for user input and AI selection
    with st.sidebar:
        query = st.text_input("Enter your research query:", "What is machine learning?")
        ai_engine = st.radio("Choose an AI engine:", ("Claude", "Mistral"))
        research_button = st.button("Start Research")

    if research_button:
        try:
            engine = ScholarSearchEngine()

            # Generate research areas
            with st.spinner("Generating research areas..."):
                research_queries = generate_research_areas(query)

            # Generate related questions
            with st.spinner("Generating related questions..."):
                research_queries_str = "\n".join(research_queries)
                related_questions = generate_queries(research_queries_str)

            # Fetch Google Scholar results
            with st.spinner("Fetching Google Scholar results..."):
                all_results = []
                for question in related_questions:
                    results = engine.fetch_google_scholar_results(question, num_results=3)
                    formatted_results = engine.format_scholar_results(results)
                    all_results.extend(formatted_results)

                web_results = json.dumps(all_results, indent=4)

            # Generate AI response
            with st.spinner(f"Generating {ai_engine} response..."):
                if ai_engine == "Mistral":
                    ai_response = mistral_scholar(query, web_results)
                else:
                    ai_response = claude_scholar(query, web_results)

            # Display AI response
            st.header(f"{ai_engine} Scholar Response")
            st.write(ai_response)

            # Dropdown for additional information
            with st.expander("View Research Details"):
                st.subheader("Research Areas")
                for i, rq in enumerate(research_queries, 1):
                    st.write(f"{i}. {rq}")

                st.subheader("Related Questions")
                for i, q in enumerate(related_questions, 1):
                    st.write(f"{i}. {q}")

                st.subheader("Sources")
                for result in all_results:
                    st.markdown(f"**Title:** {result['title']}")
                    st.markdown(f"**Source:** {result['source']}")
                    st.markdown(f"**Link:** [Click here]({result['link']})")
                    st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again or contact support if the problem persists.")

if __name__ == "__main__":
    main()
