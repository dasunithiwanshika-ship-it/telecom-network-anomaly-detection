# Telecom Network KPI Notes

## 1. Throughput

### What is Throughput?

Throughput is the amount of data successfully transmitted through a network during a specific period of time. It is commonly measured in bits per second, such as Mbps or Gbps.

### Why does it matter?

Throughput provides an indication of how much data a network can successfully handle. Low throughput may affect applications such as video streaming, file transfers and other data services.

### How can abnormal throughput indicate a network problem?

A significant or unexpected decrease in throughput may indicate congestion, poor network conditions, equipment problems, configuration issues or other network-related problems.

---

## 2. Latency

### What is Latency?

Latency is the time taken for data to travel between two points in a network. It is commonly measured in milliseconds (ms).

### Why does it matter?

Low latency is important for applications that require quick responses, including voice communication, video conferencing and interactive services.

### How can abnormal latency indicate a network problem?

An unexpected increase in latency may indicate congestion, routing problems, overloaded equipment or other network conditions that require investigation.

---

## 3. Packet Loss

### What is Packet Loss?

Packet loss occurs when packets of data transmitted across a network fail to reach their intended destination.

### Why does it matter?

Packet loss can negatively affect network applications and may result in interruptions, retransmissions, poor voice quality and unstable connections.

### How can abnormal packet loss indicate a network problem?

An unusual increase in packet loss may indicate network congestion, transmission problems, equipment faults or other network issues.

---

## 4. Jitter

### What is Jitter?

Jitter refers to variation in the arrival time of packets across a network.

### Why does it matter?

Jitter is particularly important for real-time services such as voice and video communication because inconsistent packet arrival can affect service quality.

### How can abnormal jitter indicate a network problem?

High or unusual jitter may indicate congestion, unstable network conditions or other issues affecting packet transmission.

---

## 5. Congestion

### What is Congestion?

Network congestion occurs when the amount of network traffic approaches or exceeds the available network capacity.

### Why does it matter?

Congestion can reduce network performance and affect the quality of services experienced by users.

### How can congestion affect network performance?

Congestion can contribute to increased latency, packet loss, jitter and reduced throughput.

---

## 6. Relationship Between KPIs

The KPIs are related to one another. For example, network congestion may result in increased latency and packet loss while also reducing throughput. Jitter may also increase when packet transmission becomes unstable.

Therefore, analyzing several KPIs together can provide more information about network behavior than examining a single KPI independently.

---

## 7. Relevance to Anomaly Detection

These KPIs can be used as features for analyzing network behavior. Machine learning techniques can be investigated to identify unusual patterns in the KPI measurements.

An anomaly could be a measurement or combination of measurements that differs significantly from the normal behavior observed in the dataset.

The objective of this project is to investigate whether machine learning techniques can identify such unusual patterns in telecom network KPI data.