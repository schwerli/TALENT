**classical_methods.base**
================================

Base class for classical methods implementation


Classes
~~~~~~~

.. code-block:: python

    class classical_methods(object, metaclass=abc.ABCMeta)

Abstract base class for all classical machine learning methods in TALENT.


Methods
~~~~~~~

.. code-block:: python

    classical_methods.__init__(args, is_regression)
    classical_methods.data_format(is_train=True, N=None, C=None, y=None)
    classical_methods.construct_model(model_config=None)
    classical_methods.fit(data, info, train=True, config=None)
    classical_methods.reset_stats_withconfig(config)
    classical_methods.metric(predictions, labels, y_info)

**Parameters:**

* **args** *(object)* - Configuration arguments containing model settings
* **is_regression** *(bool)* - Whether the task is regression (True) or classification (False)
* **is_train** *(bool, default=True)* - Whether processing training data or test data
* **N** *(array-like, optional)* - Numerical features data
* **C** *(array-like, optional)* - Categorical features data
* **y** *(array-like, optional)* - Target labels
* **model_config** *(dict, optional)* - Model configuration parameters
* **data** *(tuple)* - Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
* **info** *(dict)* - Dataset information
* **train** *(bool, default=True)* - Whether to train the model or just load from checkpoint
* **config** *(dict, optional)* - Additional configuration parameters
* **predictions** *(array-like)* - Model predictions
* **labels** *(array-like)* - True labels
* **y_info** *(dict)* - Label information

**Returns:**

* **time_cost** *(float)* - Training time in seconds (for fit method)
* **vres** *(tuple)* - Evaluation metrics values
* **metric_name** *(tuple)* - Names of the evaluation metrics

**Notes:**

- Abstract base class that defines the interface for all classical methods
- Handles common data preprocessing including:
  - Missing value imputation
  - Categorical encoding
  - Numerical encoding and binning
  - Data normalization
  - Label processing
- Provides unified evaluation metrics calculation
- Supports both regression and classification tasks
- Automatically handles data format conversion between training and testing
- Manages model saving and loading functionality
- For regression: returns MAE, R2, RMSE metrics
- For classification: returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics

**Abstract Methods:**

- `construct_model()`: Must be implemented by subclasses to create the specific model
- `fit()`: Must be implemented by subclasses for model training
- `predict()`: Must be implemented by subclasses for model prediction

**References:**

``[1] TALENT Framework Documentation. Classical Methods Base Class.`` 