**classical_methods.lr**
==============================

Linear Regression classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class LinearRegressionMethod(classical_methods)

Linear Regression method for regression tasks only.


Methods
~~~~~~~

.. code-block:: python

    LinearRegressionMethod.__init__(args, is_regression)
    LinearRegressionMethod.construct_model(model_config=None)
    LinearRegressionMethod.fit(data, info, train=True, config=None)
    LinearRegressionMethod.predict(data, info, model_name)
    LinearRegressionMethod.metric(predictions, labels, y_info)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Must be True for Linear Regression (regression only)
* **model_config** *(dict, optional)* - Model configuration parameters for LinearRegression
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

- Uses sklearn's LinearRegression for regression tasks only
- Requires cat_policy to be different from 'indices'
- Does not support hyperparameter tuning (tune must be False)
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- Returns MAE, R2, RMSE metrics for regression evaluation
- Linear model that assumes linear relationship between features and target
- Simple and interpretable regression method

**References:**

``[1] Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). Introduction to linear regression analysis. John Wiley & Sons.`` 