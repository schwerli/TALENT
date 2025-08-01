**classical_methods.randomforest**
========================================

Random Forest classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class RandomForestMethod(KnnMethod)

Random Forest method for classification and regression tasks using ensemble of decision trees.


Methods
~~~~~~~

.. code-block:: python

    RandomForestMethod.__init__(args, is_regression)
    RandomForestMethod.construct_model(model_config=None)
    RandomForestMethod.fit(data, info, train=True, config=None)
    RandomForestMethod.predict(data, info, model_name)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Whether the task is regression (True) or classification (False)
* **model_config** *(dict, optional)* - Model configuration parameters for Random Forest
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

- Uses sklearn's RandomForestClassifier for classification and RandomForestRegressor for regression
- Inherits from KnnMethod class for common functionality
- Supports both binary and multiclass classification
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- For regression: returns MAE, R2, RMSE metrics
- For classification: returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics
- Supports probability predictions for classification tasks
- Ensemble method that combines multiple decision trees

**References:**

``[1] Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.`` 