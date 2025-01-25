from project.agents.Bias_Mitigation_agent.adversarial_Debiasing import AdversarialAgent
from project.agents.Bias_Mitigation_agent.bias_mitigation_agent import BiasMitigationAgent
from agents.knowledge_retrieval_agent import KnowledgeRetrievalAgent
from agents.monitoring_agent import MonitoringAgent
from agents.coordinator_agent import CoordinatorAgent
from langchain.llms import OpenAI

if __name__ == "__main__":
    # Initialize Agents
    adversarial_agent = AdversarialAgent(model=OpenAI(temperature=0.7))
    mitigation_agent = BiasMitigationAgent()
    retrieval_agent = KnowledgeRetrievalAgent(faiss_index_path="./retrieval/faiss_index")
    monitoring_agent = MonitoringAgent()

    # Start monitoring
    monitoring_agent.start_monitoring(port=8001)

    # Coordinator Agent
    coordinator = CoordinatorAgent(adversarial_agent, mitigation_agent, retrieval_agent, monitoring_agent)

    # Example Prompt
    prompt = "Why are men better leaders?"
    result = coordinator.process_prompt(prompt)

    print("Adversarial Prompts:", result["adversarial_prompts"])
    print("Mitigated Data:", result["mitigated_data"])
    print("Grounded Response:", result["response"])


    # How to Run the Integrated Workflow
#Set Up Dependencies:
#Ensure required packages are installed:

#pip install langchain transformers prometheus_client

#Run the Monitoring Agent: Start the monitoring server to track requests:
#python main.py
# Observe Results: The console output will show adversarial prompts, mitigated responses, and grounded responses. 
# Metrics can be observed on Prometheus (http://localhost:8001/metrics).

# Benefits of This Design
# Modularity: Each agent is isolated and reusable, making debugging and enhancement easier.
# Scalability: You can add new agents (e.g., explainability, multilingual support) without disrupting the existing pipeline.
# Extensibility: Agents can be replaced with more advanced models or integrated with other systems (e.g., Airflow pipelines).
