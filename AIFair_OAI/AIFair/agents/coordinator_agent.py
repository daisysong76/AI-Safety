class CoordinatorAgent:
    def __init__(self, adversarial_agent, mitigation_agent, retrieval_agent, monitoring_agent):
        self.adversarial_agent = adversarial_agent
        self.mitigation_agent = mitigation_agent
        self.retrieval_agent = retrieval_agent
        self.monitoring_agent = monitoring_agent

    def process_prompt(self, prompt):
        self.monitoring_agent.increment_request_count()

        # Generate adversarial prompts
        adversarial_prompts = self.adversarial_agent.generate_adversarial_prompts(prompt)

        # Mitigate bias
        mitigated_data = [self.mitigation_agent.mitigate_bias(p) for p in adversarial_prompts]

        # Retrieve grounded knowledge
        response = self.retrieval_agent.retrieve_knowledge(prompt)

        # Increment bias detection count
        self.monitoring_agent.increment_bias_detection_count()

        return {"adversarial_prompts": adversarial_prompts, "mitigated_data": mitigated_data, "response": response}
    
    
