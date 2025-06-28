# Bati Bank - Credit Scoring Model

This project aims to develop a credit scoring model for Bati Bank's new buy-now-pay-later (BNPL) service, in partnership with a major eCommerce company. The model will leverage customer behavioral data to predict credit risk, enabling informed lending decisions.

## Project Structure

```
docker-compose.yml
Dockerfile
README.md
requirements.txt
data/
    processed/
    raw/
notebooks/
src/
    __init__.py
    data_processing.py
    predict.py
    train.py
    api/
        main.py
        pydantic_models.py
tests/
    test_data_pocessing.py
```

## Credit Scoring Business Understanding

This section outlines the core business and regulatory context that shapes our modeling approach. It addresses key questions regarding compliance, data limitations, and model selection strategy.

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord fundamentally links a bank's capital adequacy requirements to its underlying risks. For credit risk, it allows banks to use an Internal Ratings-Based (IRB) approach, where they can develop their own models to estimate risk components like Probability of Default (PD). However, to use this approach, banks must prove to regulators that their models are robust, conceptually sound, and validated.

This regulatory scrutiny directly translates into two critical needs for our model:

*   **Interpretability:** Regulators, auditors, and internal risk management committees must be able to understand *why* the model assigns a certain risk score to a customer. A "black box" model, regardless of its accuracy, is unacceptable because its decision-making logic is opaque. We must be able to explain which factors drive a customer's score and justify the model's behavior.
*   **Documentation:** Every step of the model development lifecycle—from data sourcing and feature engineering to model selection, validation, and performance monitoring—must be meticulously documented. This creates a transparent audit trail, proving that the model was built and tested according to sound statistical principles and regulatory guidelines.

In short, Basel II forces us to prioritize transparency and explainability alongside predictive power. A model is not just a predictive tool; it is a core component of the bank's risk management framework, subject to rigorous oversight.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

**Necessity of a Proxy Variable:**
Our dataset contains transactional and behavioral data from an eCommerce platform, not historical loan performance data. Since the BNPL service is new, there are no past instances of customers "defaulting" on a loan from us. A supervised machine learning model requires a target variable (a "label") to learn from. Without a direct `is_default` label, we cannot train a model to predict it.

Therefore, we must create a **proxy variable**—an observable feature that we hypothesize is highly correlated with the true, unobservable risk of default. In this project, we use customer engagement metrics (Recency, Frequency, Monetary - RFM) to identify a segment of "disengaged" customers, labeling them as `high_risk`. The underlying assumption is that customers with a weak relationship with the platform (low frequency, low spend) are more likely to default on a credit obligation.

**Potential Business Risks of Using a Proxy:**
Basing our credit decisions on a proxy variable introduces significant business risks because our core assumption might be flawed.

1.  **Risk of False Positives (Incorrectly Rejecting Good Customers):** Our proxy might label a low-frequency shopper as "high-risk." This customer might, in reality, be financially stable and would have repaid the loan. By denying them credit, we lose out on the potential revenue (interest and fees) and risk alienating a creditworthy customer, damaging our reputation.

2.  **Risk of False Negatives (Incorrectly Approving Bad Customers):** This is the more severe financial risk. A highly engaged customer (high frequency/monetary) could be over-leveraged or have poor financial habits, making them a high default risk. Our proxy would incorrectly label them as `low_risk`, leading the bank to extend credit. If this customer defaults, the bank suffers a direct **credit loss**.

3.  **Model Mismatch:** The behaviors that define a "disengaged shopper" may not be the same behaviors that define a "loan defaulter." This fundamental mismatch between the proxy and the true target can lead to a model that is systematically poor at its intended job, resulting in higher-than-expected default rates and financial losses.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

The choice between a simple and a complex model in a regulated financial context centers on the trade-off between **interpretability and performance**.

| Feature | Simple Model (e.g., Logistic Regression with WoE) | Complex Model (e.g., Gradient Boosting) |
| :--- | :--- | :--- |
| **Interpretability** | **High.** Each feature's contribution to the final score is clear and direct (via its coefficient or WoE value). It's easy to explain to regulators and business users why a specific customer received their score. This makes it ideal for generating credit scorecards. | **Low.** The model is a "black box." It's extremely difficult to trace the exact path of a single prediction through hundreds of decision trees. Explanations rely on post-hoc methods like SHAP, which are approximations and add another layer of complexity. |
| **Performance** | **Moderate.** May not capture complex, non-linear relationships between features. This can result in slightly lower predictive accuracy (e.g., a lower AUC score). | **High.** Can automatically detect and model highly complex, non-linear interactions in the data, often leading to state-of-the-art predictive performance. |
| **Regulatory Compliance**| **Easier.** Its transparency makes it straightforward to validate and get approved by regulatory bodies. The model's logic aligns well with the documentation and explainability requirements of frameworks like Basel II. | **Harder.** The lack of inherent transparency invites intense regulatory scrutiny. Proving its fairness, stability, and conceptual soundness is a significant challenge. |
| **Implementation & Maintenance** | **Simpler.** The model is less computationally expensive to train and score. The resulting scorecard is easy to implement in production systems and straightforward to maintain. | **More Complex.** Requires more computational resources, more data for robust training, and is more susceptible to overfitting. Maintenance is more challenging due to its complexity. |

**The Strategic Trade-Off for Bati Bank:**

For a regulated institution like Bati Bank, **interpretability is not a preference; it is a requirement.** While a Gradient Boosting model might offer a marginal lift in accuracy, the cost and risk associated with its regulatory approval are immense. A model that cannot be explained cannot be trusted or audited. Therefore, starting with a robust, interpretable model like **Logistic Regression with Weight of Evidence (WoE)** is the more prudent and standard industry approach. It provides a solid, defensible baseline that meets business and regulatory needs, while a more complex model could be explored later as a "challenger" model once the portfolio matures.