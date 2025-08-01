**classical_methods.dummy**
================================

Dummy (Baseline) classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class DummyMethod(classical_methods)

Dummy method for classification and regression tasks as a baseline model.


Methods
~~~~~~~

.. code-block:: python

    DummyMethod.__init__(args, is_regression)
    DummyMethod.construct_model(model_config=None)
    DummyMethod.fit(data, info, train=True, config=None)
    DummyMethod.predict(data, info, model_name)
    DummyMethod.metric(predictions, labels, y_info)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Whether the task is regression (True) or classification (False)
* **model_config** *(dict, optional)* - Model configuration parameters for Dummy model
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

- Uses sklearn's DummyClassifier for classification and DummyRegressor for regression
- DummyRegressor uses 'mean' strategy for regression tasks
- DummyClassifier uses default strategy for classification tasks
- Requires cat_policy to be different from 'indices'
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- For regression: returns MAE, R2, RMSE metrics
- For classification: returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics
- Serves as a baseline model for performance comparison

**References:**

``[1] Scikit-learn developers. (2023). sklearn.dummy.DummyClassifier. https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html`` 