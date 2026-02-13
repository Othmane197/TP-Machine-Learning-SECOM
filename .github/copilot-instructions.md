# AI Coding Agent Instructions - ML SECOM Classification Project

## Project Overview
Binary classification project on UCI SECOM dataset comparing four models: Decision Tree, Random Forest, AdaBoost, and Neural Network. Emphasis on proper handling of imbalanced data and prevention of data leakage.

## Architecture & Key Patterns

### Data Workflow (Critical)
- **SplitA approach**: Uses `imblearn.pipeline.Pipeline` (ImbPipeline) to safely apply SMOTE within each cross-validation fold
- **SplitB approach**: Manual preprocessing with separate fit/transform on train data, then transform on test
- **Essential rule**: SMOTE is NEVER applied to test data - only training data is oversampled
- **Data split**: Always use `stratify=y` in train_test_split to maintain class distribution (70% train, 30% test)

### Handling Imbalanced Data
- Target variable (`Pass/Fail`) has class imbalance: -1 (good) → 0, and 1 (defect)
- Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for cross-validation
- Apply SMOTE for oversampling minority class ON TRAINING DATA ONLY
- Evaluation metrics prioritize **Recall** and **AUC** over accuracy (imbalanced datasets)

### Model Comparison Pattern
- After CV, train final pipeline on full training set
- Evaluate on held-out test set using:
  - ROC curves with AUC scores
  - Classification reports (precision, recall, f1)
  - Confusion matrices with seaborn heatmap
  - All models use `random_state=42` for reproducibility

## File References
- [SplitA.ipynb](SplitA.ipynb): ImbPipeline approach (cleaner CV with SMOTE)
- [SplitB.ipynb](SplitB.ipynb): Manual preprocessing (explicit step control, educational)
- [uci-secom.csv](uci-secom.csv): SECOM dataset with 1567 samples, 591 features

## Common Conventions
1. **Preprocessing pipeline** order: Imputation (median) → Scaling (StandardScaler) → SMOTE (train only)
2. **Columns to drop**: 'Time' and 'Phase' removed in initial cleaning
3. **Matplotlib/Seaborn**: Always set `sns.set(style="whitegrid")` for consistent plots
4. **Comments in French**: Code comments are in French (educational context)

## When Extending This Project
- For new models, wrap them in appropriate pipeline (ImbPipeline if doing CV with SMOTE)
- Add metrics to the results dictionary: `{'pipeline': ..., 'auc': ..., 'recall': ...}`
- Always validate on stratified splits and compare against baseline (random classifier AUC=0.5)
- Test data stays untouched during ALL preprocessing adjustments
