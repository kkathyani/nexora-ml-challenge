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