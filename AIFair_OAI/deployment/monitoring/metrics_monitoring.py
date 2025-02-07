from prometheus_client import Counter, Gauge, start_http_server
import time

# Define Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Total number of requests")
BIAS_DETECTION_COUNT = Counter("bias_detection_count", "Number of bias detection calls")
RESPONSE_LATENCY = Gauge("response_latency_seconds", "Response latency in seconds")

def monitor_example():
    start_http_server(8001)  # Expose metrics at localhost:8001
    while True:
        REQUEST_COUNT.inc()  # Increment request count
        BIAS_DETECTION_COUNT.inc(3)  # Example increment
        RESPONSE_LATENCY.set(0.5)  # Set response latency
        time.sleep(5)

if __name__ == "__main__":
    monitor_example()


# Run the script:
# python deployment/monitoring/metrics_monitoring.py
# Prometheus will scrape the metrics from localhost:8001/metrics.


"""
TODO:
 Configure Grafana
Connect Prometheus as a data source in Grafana.
Create dashboards to monitor:
1. Request counts.
2. Bias detection usage.
3. Response latency trends.
"""