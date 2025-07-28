**PTaRL**
============================================================================
A regularization-based framework that enhances prediction by constructing and projecting into a prototype-based space.

**Epoch Training Function**
---------------------------

.. code-block:: python

    run_one_epoch(model, data_loader, loss_func, model_type, config, regularize, ot_weight, diversity_weight, r_weight, diversity, optimizer=None)

Runs a single training epoch, computing loss with optional regularization terms.

**Parameters:**

* **model** - The model to train/evaluate.
* **data_loader** - DataLoader providing batches of input data and labels.
* **loss_func** - Loss function (e.g., `F.cross_entropy` for classification).
* **model_type** *(str)* - Type of model; models ending with 'ot' use additional loss terms.
* **config** *(dict)* - Training configuration parameters.
* **regularize** *(bool)* - Whether to apply topic regularization.
* **ot_weight** *(float)* - Weight for optimal transport (OT) loss term.
* **diversity_weight** *(float)* - Weight for diversity loss term.
* **r_weight** *(float)* - Weight for topic regularization loss.
* **diversity** *(bool)* - Whether to apply diversity loss.
* **optimizer** *(Optional[torch.optim.Optimizer], Default is None)* - Optimizer for training (None for evaluation).

**Returns:**

* **float** - Average loss over the epoch.

**Key Features:**
- Handles mixed numerical/categorical input data.
- Adds OT loss for 'ot' model types, measuring alignment between hidden states and topics.
- Computes diversity loss by encouraging similar labels to have similar representations.
- Applies topic regularization to enforce sparse and distinct topics.


**Validation Epoch Function**
------------------------------

.. code-block:: python

    run_one_epoch_val(model, data_loader, loss_func, model_type, config, is_regression)

Runs a validation epoch and computes performance metrics.

**Parameters:**

* **model** - The model to validate.
* **data_loader** - DataLoader providing validation data.
* **loss_func** - Loss function (unused for metric calculation).
* **model_type** *(str)* - Type of model (affects prediction extraction).
* **config** *(dict)* - Configuration parameters.
* **is_regression** *(bool)* - Whether the task is regression (vs. classification).

**Returns:**

* **float** - Validation metric: accuracy (classification) or RMSE (regression).

**Metrics:**
- Classification: Accuracy (via `sklearn.metrics.accuracy_score`).
- Regression: Root mean squared error (RMSE) via `sklearn.metrics.mean_squared_error`.


**Model Fitting Function**
---------------------------

.. code-block:: python

    fit_Ptarl(args, model, train_loader, val_loader, loss_func, model_type, config, regularize, is_regression, ot_weight, diversity_weight, r_weight, diversity, seed, save_path)

Trains the model with early stopping and saves the best-performing checkpoint.

**Parameters:**

* **args** - Command-line arguments (includes `max_epoch` for training iterations).
* **model** - The model to train.
* **train_loader** / **val_loader** - DataLoaders for training and validation data.
* **loss_func** - Loss function for training.
* **model_type** *(str)* - Type of model (affects loss calculations).
* **config** *(dict)* - Training configuration (learning rate, weight decay).
* **regularize** *(bool)* - Whether to apply topic regularization.
* **is_regression** *(bool)* - Whether the task is regression.
* **ot_weight** / **diversity_weight** / **r_weight** - Weights for loss terms.
* **diversity** *(bool)* - Whether to apply diversity loss.
* **seed** *(int)* - Random seed (for checkpoint naming).
* **save_path** *(str)* - Directory to save model checkpoints.

**Returns:**

* **best_model** - The model with the best validation performance.
* **best_val_loss** - The best validation metric (accuracy/RMSE).

**Key Features:**
- Uses AdamW optimizer for training.
- Implements early stopping with patience (`early_stop=20`).
- Saves checkpoints of the best model and final model (for 'ot' models).
- Adapts early stopping logic to metric type (maximizes accuracy, minimizes RMSE).


**Testing Function**
---------------------

.. code-block:: python

    test(model, test_loader, no_ot=False)

Generates predictions on test data.

**Parameters:**

* **model** - Trained model to test.
* **test_loader** - DataLoader providing test data.
* **no_ot** *(bool, optional, Default is False)* - Whether the model is non-OT type (unused).

**Returns:**

* **pred** *(np.ndarray)* - Concatenated predictions.
* **y** *(np.ndarray)* - True labels.


**Topic Generation Function**
------------------------------

.. code-block:: python

    generate_topic(model, train_loader, n_clusters)

Generates topic centroids by clustering hidden states from the model's encoder.

**Parameters:**

* **model** - Trained model with an `encoder` attribute.
* **train_loader** - DataLoader for training data (to extract hidden states).
* **n_clusters** *(int)* - Number of topic clusters.

**Returns:**

* **cluster_centers_** *(np.ndarray)* - Centroids of the clusters (shape `(n_clusters, hidden_dim)`).

**Process:**
1. Extracts hidden states from the model's encoder for all training data.
2. Applies K-means clustering to the hidden states.
3. Returns the cluster centroids as topic representations.

##References##
.. [Ye2024PTARL] Hangting Ye, Wei Fan, Xiaozhuang Song, Shun Zheng, He Zhao, Dandan Guo, and Yi Chang. **PTARL: Prototype-based Tabular Representation Learning via Space Calibration**. In *Proceedings of the Twelfth International Conference on Learning Representations*, 2024. `<https://openreview.net/pdf?id=G32oY4Vnm8>`_