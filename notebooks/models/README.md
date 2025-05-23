# Models

This folder contains Jupyter notebooks used for training and evaluating different classification models to predict delivery risk. Each notebook includes steps for feature transformation, hyperparameter tuning, cross-validation, and model evaluation.

## Models Trained

The following classification models have been implemented and analyzed (it is recommended to check them in this order):

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Extra Trees**
5. **XGBoost**

## Methodology

Each model follows a structured workflow that includes:

- **Base model Evaluation**: A baseline model is trained with default parameters to establish a performance benchmark.
- **Hyperparameter Tuning**: Optimizing model parameters to improve performance.
- **Cross-Validation**: Evaluating model robustness using k-fold cross-validation.
- **Feature Transformation**: Scaling, encoding, or normalizing features as needed.
- **Model Evaluation**: Assessing performance using accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC.

## Best Performing Models

The model with the highest accuracy of **0.94** is:

- **Extra Trees**

This models outperformed the others in terms of accuracy and overall classification performance.

## Quick Overview of Model Performance

For a summary of the model results and insights, refer to the report:
📄 **[Will I Get My Order on Time?](../../reports/Will_I_Get_My_Order_On_Time.pdf)**

## Running Initial Data Checks

- A custom scripts have been created to train the models. To access these scripts, ensure that the `../../setup.py` file is executed first:

```bash
pip install -e .
```

Run this from the project directory, not the database folder

## Notes

- Ensure all necessary dependencies are installed before running the notebooks by running.

```bash
pip install -r requirements.txt
```

- Check `..\data_cleaning.ipynb` and `..\eda.ipynb` for more insights into the features

---
📌 For any issues or further clarification, feel free to reach out!
