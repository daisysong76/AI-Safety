from prometheus_client import Counter, start_http_server

class MonitoringAgent:
    def __init__(self):
        self.request_count = Counter("request_count", "Total number of requests")
        self.bias_detection_count = Counter("bias_detection_count", "Number of bias detection tasks performed")

    def increment_request_count(self):
        self.request_count.inc()

    def increment_bias_detection_count(self):
        self.bias_detection_count.inc()

    def start_monitoring(self, port=8001):
        start_http_server(port)
