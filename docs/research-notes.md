# Research Notes

## 1. Research Problem

Telecommunication networks continuously generate large amounts of performance and operational data. These data may contain information about network behavior, service quality and abnormal conditions.

Identifying abnormal network behavior manually can be difficult when the volume of data is large. Therefore, this project investigates whether machine learning techniques can be used to identify unusual patterns in telecom network KPI data.

## 2. Problem Statement

## 3. Research Goal

The main goal of this project is to investigate machine learning techniques for detecting anomalies in telecom network KPI data.

The project will focus on understanding the dataset, identifying relevant features, exploring normal and abnormal network behavior, and evaluating suitable machine learning approaches.

## 4. Objectives

The main objectives of the project are:

1. Understand important telecom network KPIs.
2. Obtain and understand a suitable public telecom network dataset.
3. Perform data cleaning and preprocessing.
4. Explore network KPI distributions and relationships.
5. Identify potential abnormal patterns in the data.
6. Investigate supervised anomaly classification techniques where labelled data is available.
7. Investigate unsupervised anomaly detection techniques where appropriate.
8. Compare model performance using suitable evaluation metrics.
9. Analyze important features contributing to detected anomalies.
10. Develop a small prototype for demonstrating the results.

## 5. Why Anomaly Detection?

Network performance is not always constant. KPI values can change because of traffic conditions, network configuration, equipment behavior, environmental conditions and other factors.

An anomaly is a data observation or pattern that differs significantly from expected or normal behavior.

Detecting such patterns can help identify data points that require further investigation by network engineers or operational teams.

The purpose of this project is not to automatically diagnose or repair network faults. Instead, the system will investigate whether machine learning can identify unusual network behavior that may require further investigation.

## 6. Proposed Approach

The project will follow a structured machine learning workflow:

1. Obtain a suitable public telecom network KPI dataset.
2. Understand the dataset structure and available variables.
3. Perform exploratory data analysis.
4. Clean missing, duplicated or inconsistent records where necessary.
5. Perform feature engineering where appropriate.
6. Analyze relationships between network KPIs.
7. Establish a baseline approach.
8. Investigate supervised classification if labelled anomaly data is available.
9. Investigate unsupervised anomaly detection techniques.
10. Evaluate the approaches using suitable metrics.
11. Analyze the detected anomalies.
12. Develop a small prototype or visualization of the results.

## 7. Expected Output

The expected outputs of the project are:

- A cleaned and documented telecom KPI dataset.
- Exploratory data analysis notebooks.
- Machine learning experiments.
- Anomaly detection models where appropriate.
- Evaluation results.
- Visualizations of network behavior and detected anomalies.
- Documentation of the methodology and findings.
- A small prototype demonstrating the anomaly detection process.

## 8. Project Scope

The project will initially focus on telecom network KPI data obtained from publicly available datasets.

The initial analysis will focus on metrics such as:

- Throughput
- Latency
- Packet Loss
- Jitter
- Congestion-related indicators

The project will primarily investigate data analysis and machine learning techniques rather than direct modification of telecom network infrastructure.

Any relationship between detected anomalies and real-world network faults will be treated as an area requiring further validation by telecom engineering or operational teams.

## 9. Limitations

Potential limitations include:

- The public dataset may not represent the exact network environment of the organization.
- Available KPI measurements may differ from those used in operational telecom systems.
- Some datasets may contain limited or incomplete anomaly labels.
- Machine learning detections may identify unusual patterns without identifying their exact root cause.
- Results from a public dataset may not directly represent real-world network conditions.

## 10. Future Improvements

Possible future improvements include:

- Testing the approach using real organizational network data, subject to authorization.
- Incorporating additional network KPIs.
- Including temporal and geographical information where available.
- Developing real-time or near-real-time anomaly detection.
- Adding explainability techniques to help engineers understand detected anomalies.
- Developing a monitoring dashboard.
- Integrating the system with existing network monitoring workflows.