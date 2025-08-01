**classical_methods.naivebayes**
======================================

Naive Bayes classical method implementation


Classes
~~~~~~~

.. code-block:: python

    class NaiveBayesMethod(NCMMethod)

Naive Bayes method for classification tasks using Gaussian Naive Bayes.


Methods
~~~~~~~

.. code-block:: python

    NaiveBayesMethod.__init__(args, is_regression)
    NaiveBayesMethod.construct_model(model_config=None)
    NaiveBayesMethod.fit(data, info, train=True, config=None)
    NaiveBayesMethod.predict(data, info, model_name)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Must be False for Naive Bayes (classification only)
* **model_config** *(dict, optional)* - Model configuration parameters (not used for GaussianNB)
* **data** *(tuple)* - Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
* **info** *(dict)* - Dataset information
* **train** *(bool, default=True)* - Whether to train the model or just load from checkpoint
* **config** *(dict, optional)* - Additional configuration parameters
* **model_name** *(str)* - Name of the model for saving/loading

**Returns:**

* **time_cost** *(float)* - Training time in seconds (for fit method)
* **vres** *(tuple)* - Evaluation metrics values
* **metric_name** *(tuple)* - Names of the evaluation metrics
* **test_logit** *(array-like)* - Test predictions

**Notes:**

- Uses sklearn's GaussianNB for classification tasks only
- Inherits from NCMMethod class for common functionality
- Supports both binary and multiclass classification
- Automatically handles data preprocessing including normalization and encoding
- Saves trained model to pickle file for later use
- Returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics
- Probabilistic classifier based on Bayes theorem with independence assumption
- Assumes features follow Gaussian distribution

**References:**

``[1] Rish, I. (2001). An empirical study of the naive Bayes classifier. In IJCAI 2001 workshop on empirical methods in artificial intelligence (Vol. 3, No. 22, pp. 41-46).`` 