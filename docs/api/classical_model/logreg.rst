**classical_methods.logreg**
===================================

Logistic Regression classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class LogRegMethod(classical_methods)

Logistic Regression method for classification tasks only.


Methods
~~~~~~~

.. code-block:: python

    LogRegMethod.__init__(args, is_regression)
    LogRegMethod.construct_model(model_config=None)
    LogRegMethod.fit(data, info, train=True, config=None)
    LogRegMethod.predict(data, info, model_name)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Must be False for logistic regression (classification only)
* **model_config** *(dict, optional)* - Model configuration parameters for LogisticRegression
* **data** *(tuple)* - Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
* **info** *(dict)* - Dataset information
* **train** *(bool, default=True)* - Whether to train the model or just load from checkpoint
* **config** *(dict, optional)* - Additional configuration parameters
* **model_name** *(str)* - Name of the model for saving/loading

**Returns:**

* **time_cost** *(float)* - Training time in seconds (for fit method)
* **vres** *(tuple)* - Evaluation metrics values
* **metric_name** *(tuple)* - Names of the evaluation metrics
* **test_logit** *(array-like)* - Test probability predictions

**Notes:**

- Uses sklearn's LogisticRegression for classification tasks only
- Supports both binary and multiclass classification
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- Returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics
- Always returns probability predictions for classification tasks
- Linear model that uses logistic function for probability estimation

**References:**

``[1] Hosmer Jr, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied logistic regression (Vol. 398). John Wiley & Sons.`` 