tep 1: Using Environment Variables with python-dotenv
1.1 Install python-dotenv
First, you need to install the python-dotenv library. You can do this using pip:
pip install python-dotenv
If you want to add it to your requirements.txt, you can do so by running:
echo "python-dotenv" >> requirements.txt
1.2 Create a .env File
Create a .env file in the root of your project to store your environment variables. This file should not be committed to version control (add it to your .gitignore).
touch .env
Here’s an example of what your .env file might look like:
DATABASE_URL=postgres://user:password@localhost:5432/mydatabase
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
1.3 Load Environment Variables in Your Code
In your Python code, you can load the environment variables from the .env file using python-dotenv. Here’s an example of how to do this:
print(f"Database URL: {database_url}")
# main.py or your application entry point

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
database_url = os.getenv("DATABASE_URL")
api_key = os.getenv("API_KEY")
secret_key = os.getenv("SECRET_KEY")

print(f"Database URL: {database_url}")
Step 2: Configuration Management with Helm for Kubernetes
2.1 Install Helm
If you haven't already, install Helm on your local machine. You can follow the official Helm installation guide for instructions.
2.2 Create a Helm Chart
You can create a new Helm chart for your application. Navigate to your project directory and run:
helm create my-app
This will create a new directory named my-app with a basic Helm chart structure.
2.3 Configure Values
In the my-app/values.yaml file, you can define your configuration settings for different environments. Here’s an example:
# my-app/values.yaml

replicaCount: 1

image:
  repository: my-app
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

env:
  DATABASE_URL: "postgres://user:password@localhost:5432/mydatabase"
  API_KEY: "your_api_key_here"
  SECRET_KEY: "your_secret_key_here"
2.4 Use Environment Variables in Deployment
In the my-app/templates/deployment.yaml file, you can reference the environment variables defined in values.yaml:
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
        - name: {{ .Release.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          ports:
            - containerPort: {{ .Values.service.port }}
          env:
            - name: DATABASE_URL
              value: {{ .Values.env.DATABASE_URL }}
            - name: API_KEY
              value: {{ .Values.env.API_KEY }}
            - name: SECRET_KEY
              value: {{ .Values.env.SECRET_KEY }}
2.5 Deploying with Helm
To deploy your application using Helm, run the following command:
helm install my-app ./my-app
You can also specify different values for different environments by creating separate values-<env>.yaml files (e.g., values-production.yaml, values-staging.yaml) and using the -f flag to specify the file during deployment:
helm install my-app ./my-app -f values-production.yaml
Summary
By following these steps, you will have effectively managed your environment configuration using environment variables with python-dotenv for local development and Helm for configuration management in Kubernetes. This approach helps keep sensitive information secure and allows for easy configuration changes across different environments.