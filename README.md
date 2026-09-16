# Nexora ML Challenge — Gateway Anomaly Detection

## Overview

This project implements a machine learning based anomaly detection solution for the Nexora gateway telemetry challenge.

The objective is to identify the 15 gateways most likely to require attention for each of the eight scored weeks.

The original challenge baseline uses a 3-sigma statistical anomaly detection method based on:

- `offline_duration_sec`
- `disconnection_cnt`
- `reboot_cnt`

This project extends that baseline using gateway-relative feature engineering and an Isolation Forest anomaly detection model.

The final implementation is **V4.1**.

---

## Problem

Gateway telemetry is collected hourly and contains information about:

- Connectivity
- Disconnections
- Reboots
- Network quality
- Communication performance
- System health
- Radio signal quality

The goal is not simply to classify individual telemetry records.

Instead, the system must rank gateways for each scored week so that the gateways most likely to require field attention appear in the top 15.

The challenge evaluates the submission using a cost-based scoring system rather than ordinary classification accuracy.

---

## Final Approach

The final solution combines two approaches:

### 1. 3-Sigma Baseline

The original statistical baseline is preserved in the implementation.

For each scored Monday:

1. A historical window is constructed.
2. Per-gateway mean and standard deviation are calculated.
3. The recent period is checked for values exceeding the historical mean by more than three standard deviations.
4. Gateways are ranked according to the number of anomalous hours.

The baseline uses:

```text
offline_duration_sec
disconnection_cnt
reboot_cnt
### 2. Gateway-Relative Feature Engineering

The solution uses gateway-relative behaviour to identify unusual changes in a gateway's own operating pattern.

Instead of relying only on absolute telemetry values, the features are designed to capture how a gateway behaves relative to its historical behaviour.

This helps reduce the effect of natural differences between gateways and focuses the anomaly detection process on unusual gateway-specific behaviour.

### 3. Isolation Forest

Isolation Forest is used as the machine learning anomaly detection method.

The model is suitable for this problem because the challenge does not provide a simple binary failure label for every telemetry record.

The model identifies observations that are unusual compared with the overall behaviour of the telemetry data.

The resulting anomaly information is aggregated at gateway level and used for weekly ranking.

### 4. Weekly Gateway Ranking

The final objective is to produce a ranked list of gateways rather than simply classify individual telemetry rows.

For each scored week, gateways are evaluated using their anomaly behaviour and ranked according to the resulting anomaly evidence.

The top 15 gateways are selected as the gateways requiring the most attention.

---

## Why This Approach

The approach was designed to improve on the limitations of using only absolute statistical thresholds.

The 3-sigma baseline provides a simple and interpretable reference point.

Gateway-relative features provide additional information about changes in individual gateway behaviour.

Isolation Forest provides an unsupervised machine learning approach for detecting unusual observations when labelled failure data is not directly available.

Combining these ideas allows the solution to retain the baseline as a reference while adding a machine learning based anomaly detection component.

---

## Limitations

The model should be considered an anomaly-ranking system rather than a guaranteed failure prediction system.

Important limitations include:

- An anomaly does not necessarily mean that a gateway will fail.
- Historical gateway behaviour may change over time.
- Unusual behaviour may sometimes be caused by temporary or legitimate conditions.
- The absence of a strong ground-truth failure label limits direct supervised learning.
- A top-15 ranking does not guarantee that every selected gateway will require field intervention.

---

## What It Cannot Do

The system cannot guarantee that every gateway ranked in the top 15 will experience a failure or require maintenance.

It detects and ranks unusual behaviour based on the available telemetry data.

It also cannot determine the exact physical cause of an anomaly without additional diagnostic information.

The output should therefore be treated as a decision-support ranking for identifying gateways that may deserve further investigation.

---

## Project Structure

```text
nexora-ml-challenge/
│
├── README.md
├── DECISIONS.md
├── AI-USAGE.md
├── baseline_3sigma.py
├── validate_submission.py
├── requirements.txt
├── predictions_ml_v4_1.csv
└── 23019A32J2.pdf
## Screen Recording

A 7-minute walkthrough of the Nexora ML Challenge project, covering the
problem statement, machine learning approach, implementation, and validation.

[Watch the project walkthrough](https://drive.google.com/file/d/1MIhX5w-nPO3XiRu4qvl4caCnTtfrtqXm/view?usp=sharing)