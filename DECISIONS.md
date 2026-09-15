

# NEXORA 2026 — Project Decisions

## 1. Decision: Machine Learning as the Part 2 Area

### Choice

We selected **Machine Learning** as our Part 2 focus.

The challenge provides a 3-sigma statistical baseline for identifying anomalous gateway behaviour. We chose to investigate a machine-learning-based anomaly detection approach as an extension of this baseline.

### Alternatives Considered

We considered:

- Using only the supplied statistical baseline
- Data Engineering
- Data Science
- Software Development
- MLOps

### Why We Chose Machine Learning

The main challenge is to identify and rank gateways that are most likely to require attention. Machine Learning gives us an opportunity to learn patterns across gateway telemetry rather than relying only on fixed statistical thresholds.

We therefore focused our development effort on anomaly detection and gateway-level ranking.

---

## 2. Decision: Keep the 3-Sigma Baseline

### Choice

We retained the supplied 3-sigma approach as the baseline/reference solution.

The baseline analyses:

- `offline_duration_sec`
- `disconnection_cnt`
- `reboot_cnt`

and identifies unusually high gateway activity using historical gateway behaviour.

### Alternatives Considered

We considered completely replacing the baseline with a machine-learning-only solution.

### Why We Rejected the Alternative

The supplied baseline provides a clear reference against which improvements can be evaluated.

Keeping it also allows us to understand whether the machine-learning approach provides useful additional information instead of changing the entire problem formulation.

---

## 3. Decision: Use Gateway-Relative Behaviour

### Choice

We used gateway-relative information when developing the anomaly detection approach.

### Alternatives Considered

One alternative was to use only the raw telemetry values across all gateways.

Another alternative was to use only the three variables from the supplied 3-sigma baseline.

### Why We Rejected the Alternatives

Different gateways can have different normal operating behaviour.

An absolute telemetry value may therefore be normal for one gateway but unusual for another.

Using gateway-relative information allows the solution to consider deviations from a gateway's historical behaviour rather than treating every gateway identically.

---

## 4. Decision: Use Isolation Forest for Machine-Learning Anomaly Detection

### Choice

We selected **Isolation Forest** for the machine-learning anomaly detection component.

### Alternatives Considered

We considered supervised classification approaches such as:

- Logistic Regression
- Random Forest
- Gradient Boosting

### Why We Rejected the Alternatives

The challenge does not provide a simple labelled target for every telemetry record indicating whether that observation represents a genuine gateway failure.

A supervised classifier would therefore require assumptions or constructed labels.

Isolation Forest is an unsupervised anomaly-detection method and is better aligned with the available problem formulation, where the objective is to identify unusual gateway behaviour.

---

## 5. Decision: Rank Gateways Instead of Individual Records

### Choice

We produce a **gateway-level ranking for each scored week** rather than simply returning individual anomalous telemetry records.

The operational objective is to identify the gateways that should receive field attention.

### Alternatives Considered

We could have returned:

- Individual anomalous hourly records
- A binary anomaly flag for each telemetry record
- A list of all gateways without ranking

### Why We Rejected the Alternatives

The challenge requires a limited number of gateways to be selected for field attention.

A gateway-level ranking directly represents that operational decision and allows multiple anomalous observations from the same gateway to contribute to its overall priority.

---

# What It Cannot Do

The solution should be interpreted as an anomaly-ranking system, not as a guaranteed gateway-failure predictor.

### 1. Anomaly does not guarantee failure

An anomalous telemetry pattern does not necessarily mean that a gateway is actually faulty.

Temporary network conditions or other short-lived events can produce unusual measurements.

### 2. Historical behaviour can change

The solution relies on historical telemetry to understand gateway behaviour.

If the underlying operating conditions change significantly, historical patterns may become less representative of current behaviour.

### 3. No perfect ground-truth failure label

The available challenge data does not provide a simple ground-truth label for every telemetry record indicating whether a field visit was definitely required.

Therefore, anomaly detection is being used as a proxy for identifying gateways that deserve attention.

### 4. Ranking does not guarantee the optimal field decision

The final ranking is an analytical recommendation. It does not guarantee that every selected gateway will require an engineer visit.

Operational information that is not present in the telemetry could change the correct decision.

---

# What We Would Improve With More Time

With additional development time, we would:

1. Perform more systematic validation using historical field-visit information.
2. Test multiple anomaly-scoring and ranking strategies.
3. Evaluate the effect of different feature combinations.
4. Investigate changes in gateway behaviour over time.
5. Improve the explanation/reason field associated with each ranked gateway.
6. Evaluate the final ranking more directly against the challenge's operational cost function.

---

# Summary

Our overall approach was to retain the supplied 3-sigma solution as a transparent reference while investigating whether machine-learning-based anomaly detection and gateway-relative behaviour could provide additional information for gateway prioritisation.

The final objective is not simply to detect anomalous telemetry records, but to produce a useful ranked list of gateways for each scored week.
