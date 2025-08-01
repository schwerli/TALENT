**classical_methods.lightgbm**
====================================

LightGBM classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class LightGBMMethod(XGBoostMethod)

LightGBM method for classification and regression tasks using gradient boosting.


Methods
~~~~~~~

.. code-block:: python

    LightGBMMethod.__init__(args, is_regression)
    LightGBMMethod.construct_model(model_config=None)
    LightGBMMethod.fit(data, info, train=True, config=None)
    LightGBMMethod.predict(data, info, model_name)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Whether the task is regression (True) or classification (False)
* **model_config** *(dict, optional)* - Model configuration parameters for LightGBM
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

- Uses LightGBM's LGBMClassifier for classification and LGBMRegressor for regression
- Inherits from XGBoostMethod class for common functionality
- Requires cat_policy to be different from 'indices'
- Supports early stopping with validation set evaluation
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- For regression: returns MAE, R2, RMSE metrics
- For classification: returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics
- Supports probability predictions for classification tasks
- High-performance gradient boosting framework

**References:**

``[1] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30.`` 