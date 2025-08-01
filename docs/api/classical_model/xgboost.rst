**classical_methods.xgboost**
====================================

XGBoost classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class XGBoostMethod(classical_methods)

XGBoost method for classification and regression tasks using gradient boosting.


Methods
~~~~~~~

.. code-block:: python

    XGBoostMethod.__init__(args, is_regression)
    XGBoostMethod.construct_model(model_config=None)
    XGBoostMethod.fit(data, info, train=True, config=None)
    XGBoostMethod.predict(data, info, model_name)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Whether the task is regression (True) or classification (False)
* **model_config** *(dict, optional)* - Model configuration parameters for XGBoost
* **data** *(tuple)* - Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
* **info** *(dict)* - Dataset information
* **train** *(bool, default=True)* - Whether to train the model or just load from checkpoint
* **config** *(dict, optional)* - Additional configuration parameters
* **model_name** *(str)* - Name of the model for saving/loading

**Returns:**

* **time_cost** *(float)* - Training time in seconds (for fit method)
* **vres** *(tuple)* - Evaluation metrics values
* **metric_name** *(tuple)* - Names of the evaluation metrics
* **test_logit** *(array-like)* - Test predictions (probabilities for classification, values for regression)

**Notes:**

- Uses XGBoost's XGBClassifier for classification and XGBRegressor for regression
- Supports early stopping with validation set evaluation
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- For regression: returns MAE, R2, RMSE metrics
- For classification: returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics
- Supports probability predictions for classification tasks

**References:**

``[1] Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining (pp. 785-794).`` 