# Model Quality & Classification Performance Report

Based on my analysis of the `votingclassifier_report.txt` and the `normalized_cm_votingclassifier.png` in the `outputs/` directory, here is the breakdown of how well the model is currently performing for each class.

## Summary of Findings

The model is **highly accurate at detecting "No DR"** but exhibits a **significant bias towards the "Moderate" label**. This explains why many of your tests (especially those that might actually be Mild, Severe, or Proliferative) are returning a "Moderate" result.

### 📊 Performance by Class

| Class # | Diagnosis | Recall (Detection Rate) | Accuracy Note |
| :--- | :--- | :--- | :--- |
| **0** | **No DR** | **98%** | **Near perfect**. Excellent at identifying healthy retinas. |
| **1** | **Mild** | **40%** | **Low**. 40% are detected correctly, but many are misclassified as Moderate. |
| **2** | **Moderate** | **80%** | **Strong**. Correctly identifies the majority of moderate cases. |
| **3** | **Severe** | **8%** | **Critical Issue**. Nearly all severe cases are misclassified as Moderate. |
| **4** | **Proliferative** | **12%** | **Critical Issue**. Most advanced cases are misclassified as Moderate. |

## ⚠️ Known Bias: The "Moderate" Trap

The confusion matrix shows that the model tends to use **Moderate (Class 2)** as a "catch-all" for any sign of Diabetic Retinopathy.

> [!WARNING]
> - **Mild Cases**: Over 38% of Mild images are misclassified as Moderate.
> - **Advanced Cases**: Over 65% of Proliferative images are misclassified as Moderate.

### Why is this happening?
This usually occurs due to **class imbalance** in the training dataset or overlapping feature signatures between classes. In this model, the "Moderate" class has much stronger internal feature representations than the "Mild" or "Severe" classes.

## Recommendations for Improvement

1.  **Data Rebalancing**: Augment the training data specifically for Class 1 (Mild), Class 3 (Severe), and Class 4 (Proliferative).
2.  **Fine-Tuning**: Adjust the decision thresholds for the Voting Classifier to be more sensitive to the features of "Mild" and "Severe" cases.
3.  **Ensemble Weighting**: Increase the weight of models that specifically perform well on the minority classes.

---
*Report Generated: 2026-03-30*
