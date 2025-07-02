from langgraph.graph import StateGraph
from src.langgraphagenticai.state.state import State
from langgraph.graph import START,END
from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
import asyncio
from dotenv import load_dotenv
load_dotenv()

class GraphBuilder:
    def __init__(self,model,user_controls_input,message):
        self.llm=model
        self.user_controls_input=user_controls_input
        self.message=message
        self.current_llm=user_controls_input["selected_llm"]
        self.graph_builder=StateGraph(State)  #StateGraph is a class in LangGraph that is used to build the graph

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """
        self.basic_chatbot_node=BasicChatbotNode(self.llm)

        self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)
    
    def setup_graph(self,usecase:str):
        """
        Sets up the graph for the selected use case.
        """

        self.basic_chatbot_build_graph()

        return self.graph_builder.compile()
