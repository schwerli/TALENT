**classical_methods.ncm**
==============================

Nearest Centroid Method (NCM) classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class NCMMethod(classical_methods)

Nearest Centroid Method for classification tasks using centroid-based classification.


Methods
~~~~~~~

.. code-block:: python

    NCMMethod.__init__(args, is_regression)
    NCMMethod.construct_model(model_config=None)
    NCMMethod.fit(data, info, train=True, config=None)
    NCMMethod.predict(data, info, model_name)
    NCMMethod.metric(predictions, labels, y_info)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Must be False for NCM (classification only)
* **model_config** *(dict, optional)* - Model configuration parameters (not used for NearestCentroid)
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

- Uses sklearn's NearestCentroid for classification tasks only
- Requires cat_policy to be different from 'indices'
- Does not support hyperparameter tuning (tune must be False)
- Supports both binary and multiclass classification
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- Returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics
- Distance-based classifier that finds the nearest class centroid
- Simple and interpretable classification method

**References:**

``[1] Tibshirani, R., Hastie, T., Narasimhan, B., & Chu, G. (2002). Diagnosis of multiple cancer types by shrunken centroids of gene expression. Proceedings of the National Academy of Sciences, 99(10), 6567-6572.`` 