**classical_methods.knn**
==============================

K-Nearest Neighbors (KNN) classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class KnnMethod(classical_methods)

K-Nearest Neighbors method for classification and regression tasks.


Methods
~~~~~~~

.. code-block:: python

    KnnMethod.__init__(args, is_regression)
    KnnMethod.construct_model(model_config=None)
    KnnMethod.fit(data, info, train=True, config=None)
    KnnMethod.predict(data, info, model_name)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Whether the task is regression (True) or classification (False)
* **model_config** *(dict, optional)* - Model configuration parameters for KNN
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

- Uses sklearn's KNeighborsClassifier for classification and KNeighborsRegressor for regression
- Supports both binary and multiclass classification
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- For regression: returns MAE, R2, RMSE metrics
- For classification: returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics
- Supports probability predictions for classification tasks
- Distance-based algorithm that finds k nearest neighbors for prediction

**References:**

``[1] Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE transactions on information theory, 13(1), 21-27.`` 