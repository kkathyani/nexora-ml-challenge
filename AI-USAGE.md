# AI Usage

## Overview

AI-assisted development tools were used during the development of this project.

AI assistance was used as a development and review aid for understanding the challenge, exploring machine learning approaches, reviewing feature engineering, debugging code, checking implementation decisions, and preparing documentation.

The final implementation was reviewed, executed, and validated by the project team.

---

## Areas Where AI Was Used

### 1. Understanding the Challenge

AI assistance was used to understand:

- The gateway anomaly detection problem
- The supplied 3-sigma baseline
- The cost-based evaluation objective
- The required prediction format
- The importance of detecting faulty gateways early
- The need to evaluate the model on unseen and later data

---

### 2. Machine Learning Approach

AI assistance was used to explore suitable anomaly detection approaches for the telemetry data.

The approaches discussed included:

- Statistical anomaly detection
- Isolation Forest
- Gateway-relative anomaly detection
- Historical versus recent behavior analysis

The final implementation uses an **Isolation Forest** model together with gateway-relative feature engineering.

---

### 3. Feature Engineering

AI assistance was used to review the available telemetry features and organize them into meaningful groups, including:

- Connectivity features
- Reboot features
- Communication features
- Network and radio-quality features
- System-health features

Gateway-relative features were also developed using historical and recent telemetry statistics, including:

- Mean
- Standard deviation
- Maximum
- Median
- Deviation
- Ratio
- Z-score
- Maximum deviation
- Maximum z-score

---

### 4. Temporal Data Handling

AI assistance was used to review the model's temporal data design and identify a mismatch between the intended and implemented historical window.

The final V4.1 implementation uses:

```text
Historical period:
Monday - 35 days → Monday - 7 days

Recent period:
Monday - 7 days → Monday