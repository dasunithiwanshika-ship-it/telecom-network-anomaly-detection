# Viva Voce & Technical Defense Preparation Guide
## Frequently Asked Questions & Defensible Answers

**Project:** AI-Based Network Anomaly & Fault-Risk Detection for Telecom Networks  
**Dataset:** Hugging Face `AliMaatouk/TelecomTS`  

---

### Q1: Why did you summarize 128 timesteps into 4 statistical moments rather than feeding raw time series into an LSTM or CNN?
**Answer:**  
In cellular operations, computing 4 summary statistics ($\mu, \sigma, \min, \max$) across short observation windows (~12.7 seconds) provides an ultra-low-latency tabular representation ($<0.2\text{ ms}$ inference) suitable for real-time edge or base-station telemetry. It creates scale-invariant features for decision trees and establishes a clean, interpretable baseline before introducing more complex, computationally expensive sequential deep learning architectures.

---

### Q2: Why is accuracy an unsuitable metric for this research?
**Answer:**  
The TelecomTS dataset is heavily imbalanced: 96.14% of the traces are normal, and only 3.86% are anomalous. A trivial dummy model predicting "Normal" for every trace achieves 96.14% accuracy while failing to detect a single network fault. Therefore, we evaluate using Precision-Recall Area Under the Curve (PR-AUC), F1-score, and precision/recall, which directly penalize false alarms and missed anomalies on the minority class.

---

### Q3: Why does Random Forest achieve 100% precision with zero false positives on the test set while Logistic Regression produced 9 false alarms?
**Answer:**  
Logistic Regression is a linear generalized linear model that forms an additive linear decision boundary $\mathbf{w}^T\mathbf{x} + b$. In traces with heavy but legitimate user traffic (e.g., high PRB utilization and large byte volume), positive additive contributions push the linear dot product past the threshold, triggering false alarms. Random Forest partitions feature space orthogonally using hierarchical decision thresholds, enabling it to distinguish high-traffic healthy connections from genuine faults where high PRB utilization is accompanied by degraded SNR and elevated BLER.

---

### Q4: Why did unsupervised models (Isolation Forest and Local Outlier Factor) perform so poorly (F1 < 30%) compared to Random Forest (F1 = 96.44%)?
**Answer:**  
Unsupervised anomaly detectors operate on geometric isolation or density in feature space without label guidance. In cellular networks, legitimate user behaviors (e.g. video streaming, massive file downloads) create sparse, high-dimensional observations that look like "outliers" to an unsupervised model, generating 174–193 false alarms. Furthermore, many genuine network faults (such as handover oscillation or gradual buffer buildup) occur within the normal operating range of individual KPIs and are only distinguishable through specific multi-variable combinations that require supervised labels to learn.

---

### Q5: How did you ensure a fair comparison between supervised and unsupervised models?
**Answer:**  
First, unsupervised models were fitted strictly on unlabelled training data ($X_{\text{train}}$) without any access to ground-truth labels ($y$). Second, all four models were evaluated on the exact same reserved 80/20 test split ($N=6,400$). Third, unsupervised decision scores were inverted so that higher values consistently represent greater anomaly likelihood, enabling identical ROC and Precision-Recall curve derivations across all four paradigms.

---

### Q6: Can you explain Sample 1047? Why did Logistic Regression detect it while Random Forest missed it?
**Answer:**  
Sample 1047 is an uplink buffer overflow anomaly during a Twitch streaming session. The trace had an elevated buffer (`Estimated_UL_Buffer_mean = 708.01 KB`), placing it in the 86th percentile of buffer sizes. Because Logistic Regression assigned a large standardized positive coefficient ($\beta \approx +3.65$) to buffer size, this single elevated feature pushed the linear probability to $P=1.0000$. In Random Forest, exactly 40 out of 100 trees voted anomaly, yielding an ensemble probability of $\hat{P}=0.4000$. Because $0.4000 < 0.50$ (the default decision threshold), the ensemble classified it as Normal. Lowering the threshold to $\tau \le 0.40$ allows Random Forest to capture Sample 1047 as well.

---

### Q7: What are the 16 mutually missed anomalies ($\mathcal{FN}_{\text{both}} = 16$)?
**Answer:**  
These 16 traces correspond primarily to rapid temporal anomalies, such as `Faulty Handover Algorithm (Too Frequent)`. Because our feature engineering compresses 128 timesteps into static summary moments ($\mu, \sigma, \min, \max$), the sequential ordering of rapid ping-pong handover transitions is lost. The summary statistics of these traces overlap almost completely with normal operating ranges, making them undetectable from windowed summary moments alone.

---

### Q8: What are the limitations regarding SLTMobitel?
**Answer:**  
We must be completely clear: this prototype was developed and evaluated strictly using the public `AliMaatouk/TelecomTS` cellular benchmark. It does NOT use proprietary SLTMobitel operational telemetry, does not represent Sri Lankan telecommunications infrastructure, and cannot predict physical DP/FDP cable cuts or dispatch physical repair crews. It is an algorithmic research study on cellular KPI anomaly detection.
