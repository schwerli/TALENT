**classical_methods.catboost**
====================================

CatBoost classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class CatBoostMethod(classical_methods)

CatBoost method for classification and regression tasks with native categorical feature support.


Methods
~~~~~~~

.. code-block:: python

    CatBoostMethod.__init__(args, is_regression)
    CatBoostMethod.fit(data, info, train=True, config=None)
    CatBoostMethod.predict(data, info, model_name)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Whether the task is regression (True) or classification (False)
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

- Uses CatBoost's CatBoostClassifier for classification and CatBoostRegressor for regression
- Requires cat_policy to be 'indices' for categorical feature handling
- Native support for categorical features without preprocessing
- Supports early stopping with validation set evaluation
- Automatically handles data preprocessing for numerical features
- Saves trained model to pickle file for later use
- For regression: returns MAE, R2, RMSE metrics
- For classification: returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics
- Supports probability predictions for classification tasks

**References:**

``[1] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. Advances in neural information processing systems, 31.`` 