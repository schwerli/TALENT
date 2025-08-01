**classical_methods.svm**
===============================

Support Vector Machine (SVM) classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class SvmMethod(classical_methods)

Support Vector Machine method for classification and regression tasks.


Methods
~~~~~~~

.. code-block:: python

    SvmMethod.__init__(args, is_regression)
    SvmMethod.construct_model(model_config=None)
    SvmMethod.fit(data, info, train=True, config=None)
    SvmMethod.predict(data, info, model_name)
    SvmMethod.metric(predictions, labels, y_info)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Whether the task is regression (True) or classification (False)
* **model_config** *(dict, optional)* - Model configuration parameters
* **data** *(tuple)* - Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
* **info** *(dict)* - Dataset information
* **train** *(bool, default=True)* - Whether to train the model or just load from checkpoint
* **config** *(dict, optional)* - Additional configuration parameters
* **model_name** *(str)* - Name of the model for saving/loading
* **predictions** *(array-like)* - Model predictions
* **labels** *(array-like)* - True labels
* **y_info** *(dict)* - Label information

**Returns:**

* **time_cost** *(float)* - Training time in seconds (for fit method)
* **vres** *(tuple)* - Evaluation metrics values
* **metric_name** *(tuple)* - Names of the evaluation metrics
* **test_logit** *(array-like)* - Test predictions

**Notes:**

- Uses sklearn's LinearSVC for classification and LinearSVR for regression
- Supports both binary and multiclass classification
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- For regression: returns MAE, R2, RMSE metrics
- For classification: returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics

**References:**

``[1] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.`` 