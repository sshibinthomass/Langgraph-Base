from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
import sys
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import asyncio #asyncio is a library for asynchronous programming in Python.
from pathlib import Path
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))
from src.langgraphagenticai.state.state import State
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")



class BasicChatbotNode:
    """
    Basic Chatbot login implementation
    """
    def __init__(self,model):
        self.llm=model

    def process(self,state:State)->dict:
        """
        Processes the input state and generates a chatbot response.
        """
        response = self.llm.invoke(state['messages'])
    
    #Error handling for the response 
        # If response is an AIMessage, extract .content
        if hasattr(response, "content"):
            return {"messages": response.content}
        # If response is a dict with 'content', extract it
        if isinstance(response, dict) and 'content' in response:
            return {"messages": response['content']}
        # If response is a string, return as is
        return {"messages": response}


if __name__ == "__main__":
    

    # Create LLM instance
    llm = ChatGroq(model="qwen-qwq-32b")
    
    # Create RestaurantRecommendationNode instance with the LLM
    node = BasicChatbotNode(llm)
    
    # Example conversation history
    search_state = {
        "messages": [
            SystemMessage(content="You are a helpful and efficient assistant. You help the user find the best sushi restaurants in Munich using up-to-date weather information and Google reviews. You also help the user find the best parking spots in Munich, considering current conditions. Always provide accurate, relevant, and concise recommendations."),
            HumanMessage(content="Hi"),
            AIMessage(content="Hello, I'm here to help. How can I assist you today? Are you looking for recommendations on sushi restaurants in Munich or perhaps assistance with finding a parking spot?"),
            HumanMessage(content="Yes, what are the available restaurants?"),
            AIMessage(content="Here are some sushi restaurants in Munich:\n\n1. Sasou - Address: Marienplatz 28, 80331 Munich, Food Types: Japanese - Sushi, Asian, Japanese, Southeast Asian, German, Fast Food, Soup, Price Level: 3 out of 5\n2. Galeria Restaurant - Address: Kaufingerstraße 1, 80331 Munich, Food Types: Japanese - Sushi, European, German, Price Level: 1 out of 5\n3. Shaokao Asian Grill&Wine - Address: Fürstenfelder Straße 9, 80331 Munich, Food Types: Japanese - Sushi, Asian, Chinese, Japanese, European, International, Fusion, Grill, Price Level: Unknown\n4. Secret Garden - Address: Heiliggeiststraße 2A, 80331 Munich, Food Types: Japanese - Sushi, Asian, Vietnamese, Vegan, Price Level: Unknown\n\nLet me know if you need more information or if you'd like to get reviews for any of these restaurants."),
            HumanMessage(content="Can you give google reviews for Galeria Restaurant"),
            AIMessage(content="The Galeria Restaurant has an average rating of 3.7 out of 5 based on 63 Google reviews. The restaurant offers a wide variety of food, including sushi, and has a comfortable seating area with a view of the church. However, some reviewers have mentioned that the food can be expensive, especially when buying by weight, and that the labeling of food items can be unclear. The staff has also been described as unfriendly and rude by some reviewers. Overall, the restaurant seems to be a good option for those looking for a variety of food options in a convenient location, but may not be the best choice for those on a budget or with specific dietary restrictions."),
            HumanMessage(content="What does people tell about reviews?"),
            AIMessage(content="Based on the Google reviews, people have mixed opinions about the Galeria Restaurant. Some reviewers have praised the restaurant's variety of food, comfortable seating, and convenient location. However, others have mentioned that the food can be expensive, especially when buying by weight, and that the labeling of food items can be unclear. Some reviewers have also described the staff as unfriendly and rude. Overall, the restaurant seems to be a good option for those looking for a variety of food options in a convenient location, but may not be the best choice for those on a budget or with specific dietary restrictions."),
        ]
    }

    # Call the search_node method and print the result
    search_result = node.evaluate_node(search_state)
    print("Search Node Result:", search_result)
