**HyperFast**
=========================================

A meta-trained hypernetwork that generates task-specific neural networks for instant classification of tabular data.



class HyperFast(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~

A neural network model using hypernetworks to generate weights for the main network, with support for random features and PCA preprocessing.

.. code-block:: python

    __init__(self, cfg)

**Parameters:**

* **cfg** - Configuration object with model parameters:
  - `n_dims`: Number of dimensions after PCA.
  - `max_categories`: Maximum number of categories for one-hot encoding.
  - `rf_size`: Size of random feature layer.
  - `torch_pca`: Whether to use PyTorch-based PCA.
  - `clip_data_value`: Value to clip input data to.
  - `hn_n_layers`: Number of layers in hypernetworks.
  - `hn_hidden_size`: Hidden size of hypernetworks.
  - `main_n_layers`: Number of layers in the main network.


.. code-block:: python

    forward(self, X, y, n_classes)

Forward pass of the HyperFast model, including random feature generation, PCA, and hypernetwork weight generation.

**Parameters:**

* **X** *(torch.Tensor)* - Input features, shape `(batch_size, ...)`.
* **y** *(torch.Tensor)* - Labels, shape `(batch_size,)`.
* **n_classes** *(int)* - Number of output classes.

**Returns:**

* **rf** - Random feature layer.
* **self.pca** - PCA instance used for preprocessing.
* **main_network** - List of linear layers in the main network.


**Utility Functions**
---------------------

.. code-block:: python

    seed_everything(seed: int)

Sets random seeds for reproducibility across libraries.

**Parameters:**

* **seed** *(int)* - Seed value for random number generators.


.. code-block:: python

    nn_bias_logits(test_logits, test_samples, train_samples, train_labels, bias_param, n_classes)

Adjusts test logits using a nearest neighbor bias term.

**Parameters:**

* **test_logits** *(torch.Tensor)* - Test logits, shape `(n_test, n_classes)`.
* **test_samples** *(torch.Tensor)* - Test samples, shape `(n_test, features)`.
* **train_samples** *(torch.Tensor)* - Training samples, shape `(n_train, features)`.
* **train_labels** *(torch.Tensor)* - Training labels, shape `(n_train,)`.
* **bias_param** - Bias parameter to add to logits.
* **n_classes** *(int)* - Number of classes.

**Returns:**

* **torch.Tensor** - Adjusted test logits.


.. code-block:: python

    forward_main_network(x, main_network)

Forward pass through the main network with residual connections.

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor.
* **main_network** - List of linear layers (matrix and bias tuples).

**Returns:**

* **x** *(torch.Tensor)* - Output tensor.
* **intermediate_activations** *(torch.Tensor)* - Activations from the second-to-last layer.


.. code-block:: python

    svd_flip(u, v, u_based_decision=True)

Sign correction for SVD to ensure deterministic output.

**Parameters:**

* **u** *(torch.Tensor)* - Left singular vectors.
* **v** *(torch.Tensor)* - Right singular vectors.
* **u_based_decision** *(bool, optional, Default is True)* - Whether to base sign decisions on `u`.

**Returns:**

* **u** *(torch.Tensor)* - Corrected left singular vectors.
* **v** *(torch.Tensor)* - Corrected right singular vectors.


**PCA Classes**
---------------

class TorchPCA
~~~~~~~~~~~~~~

PyTorch-based PCA implementation for dimensionality reduction.

.. code-block:: python

    __init__(self, n_components=None, fit="full")

**Parameters:**

* **n_components** *(Optional[int], Default is None)* - Number of components to keep (None = min(n_samples, n_features)).
* **fit** *(str, optional, Default is "full")* - SVD method: "full" (full SVD) or "lowrank" (low-rank SVD).


.. code-block:: python

    fit(self, X)

Fits PCA to input data.

**Parameters:**

* **X** *(torch.Tensor)* - Input data, shape `(n_samples, n_features)`.

**Returns:**

* **self** - Fitted PCA instance.


.. code-block:: python

    transform(self, X)

Applies dimensionality reduction to input data.

**Parameters:**

* **X** *(torch.Tensor)* - Input data, shape `(n_samples, n_features)`.

**Returns:**

* **torch.Tensor** - Transformed data, shape `(n_samples, n_components)`.


.. code-block:: python

    fit_transform(self, X)

Fits PCA and transforms input data in one step.

**Parameters:**

* **X** *(torch.Tensor)* - Input data, shape `(n_samples, n_features)`.

**Returns:**

* **torch.Tensor** - Transformed data, shape `(n_samples, n_components)`.


**Trainable Main Network**
--------------------------

class MainNetworkTrainable(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A trainable wrapper for the main network, including random features, PCA, and linear layers.

.. code-block:: python

    __init__(self, cfg, n_classes, rf, pca, main_network, nn_bias)

**Parameters:**

* **cfg** - Configuration object.
* **n_classes** *(int)* - Number of output classes.
* **rf** - Random feature layer.
* **pca** - PCA instance.
* **main_network** - List of linear layers (matrix and bias tuples).
* **nn_bias** - Nearest neighbor bias parameters.


.. code-block:: python

    forward(self, X, y=None)

Forward pass of the trainable main network.

**Parameters:**

* **X** *(torch.Tensor)* - Input features.
* **y** *(Optional[torch.Tensor], Default is None)* - Labels (required for bias adjustment).

**Returns:**

* **torch.Tensor** - Output logits.


.. code-block:: python

    get_main_network_parts(self)

Retrieves reconstructed components of the main network.

**Returns:**

* **rf_reconstructed** - Random feature layer.
* **pca_reconstructed** - PCA instance with learned parameters.
* **main_network_reconstructed** - List of linear layers.
* **self.nn_bias** - Nearest neighbor bias parameters.


**Fine-Tuning Function**
------------------------

.. code-block:: python

    fine_tune_main_network(cfg, X, y, n_classes, rf, pca, main_network_layers, nn_bias, device, optimize_steps, batch_size)

Fine-tunes the main network using backpropagation.

**Parameters:**

* **cfg** - Configuration object.
* **X** *(torch.Tensor)* - Training features.
* **y** *(torch.Tensor)* - Training labels.
* **n_classes** *(int)* - Number of classes.
* **rf** - Random feature layer.
* **pca** - PCA instance.
* **main_network_layers** - List of linear layers.
* **nn_bias** - Nearest neighbor bias parameters.
* **device** - Target device (e.g., "cuda" or "cpu").
* **optimize_steps** *(int)* - Number of optimization steps.
* **batch_size** *(int)* - Batch size for training.

**Returns:**

* **Tuple** - Reconstructed network components (rf, pca, main network, bias).


**Hypernetwork Helpers**
------------------------

.. code-block:: python

    get_main_weights(x, hn, weight_gen=None)

Generates weights for the main network using a hypernetwork.

**Parameters:**

* **x** *(torch.Tensor)* - Input to the hypernetwork.
* **hn** - Hypernetwork module.
* **weight_gen** *(Optional[nn.Module], Default is None)* - Linear layer to generate weights.

**Returns:**

* **torch.Tensor** - Generated weights.


.. code-block:: python

    forward_linear_layer(x, w, hs)

Applies a linear layer using generated weights.

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor.
* **w** *(torch.Tensor)* - Weights (including bias).
* **hs** *(int)* - Hidden size (number of output features).

**Returns:**

* **x** *(torch.Tensor)* - Output tensor after linear transformation.
* **(m, b)** - Tuple of weight matrix and bias vector.


.. code-block:: python

    transform_data_for_main_network(X, cfg, rf, pca)

Transforms input data for the main network (random features + PCA).

**Parameters:**

* **X** *(torch.Tensor)* - Input features.
* **cfg** - Configuration object.
* **rf** - Random feature layer.
* **pca** - PCA instance.

**Returns:**
* **torch.Tensor** - Transformed data.


**Clustering and Nearest Neighbor Utilities**
---------------------------------------------

.. code-block:: python

    distance_matrix(x, y=None, p=2)

Computes the distance matrix between two sets of points.

**Parameters:**

* **x** *(torch.Tensor)* - First set of points, shape `(n, d)`.
* **y** *(Optional[torch.Tensor], Default is None)* - Second set of points (defaults to `x`), shape `(m, d)`.
* **p** *(int, optional, Default is 2)* - Norm order (e.g., 2 for Euclidean distance).

**Returns:**

* **torch.Tensor** - Distance matrix, shape `(n, m)`.


class NN
~~~~~~~~

Nearest Neighbor classifier.

.. code-block:: python

    __init__(self, X=None, Y=None, p=2)

**Parameters:**

* **X** *(Optional[torch.Tensor], Default is None)* - Training samples.
* **Y** *(Optional[torch.Tensor], Default is None)* - Training labels.
* **p** *(int, optional, Default is 2)* - Norm order for distance calculation.


.. code-block:: python

    train(self, X, Y)

Trains the nearest neighbor classifier.

**Parameters:**

* **X** *(torch.Tensor)* - Training samples, shape `(n_train, features)`.
* **Y** *(torch.Tensor)* - Training labels, shape `(n_train,)`.


.. code-block:: python

    predict(self, x, mini_batches=True)

Predicts labels for input samples.

**Parameters:**

* **x** *(torch.Tensor)* - Input samples, shape `(n_test, features)`.
* **mini_batches** *(bool, optional, Default is True)* - Whether to process in mini-batches.

**Returns:**

* **torch.Tensor** - Predicted labels, shape `(n_test,)`.


.. code-block:: python

    predict_from_training_with_LOO(self, mini_batches=True)

Predicts labels for training samples using leave-one-out (excludes self from neighbors).

**Parameters:**

* **mini_batches** *(bool, optional, Default is True)* - Whether to process in mini-batches.

**Returns:**

* **torch.Tensor** - Predicted labels, shape `(n_train,)`.

**References:**
HyperFast: Instant Classification for Tabular Data
David Bonet, Daniel Mas Montserrat, Xavier Giró-i-Nieto, Alexander G. Ioannidis
arXiv preprint arXiv:2402.14335, 2024.

PDF `<https://arxiv.org/abs/2402.14335>`_