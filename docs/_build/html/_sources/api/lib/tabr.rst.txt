**TaBR**
=================================================

A deep learning model that integrates a KNN component to enhance tabular data predictions through an efficient attention-like mechanism.


**Embedding Utilities**
------------------------

.. code-block:: python

    def _initialize_embeddings(weight: Tensor, d: Optional[int]) -> None

Initializes embedding weights using a uniform distribution scaled by the reciprocal square root of the dimension.

**Parameters:**

* **weight** *(Tensor)* - Embedding weight tensor to initialize.
* **d** *(Optional[int])* - Dimension for scaling (defaults to `weight.shape[-1]`).

**Returns:**

* **None** - Modifies `weight` in-place.


.. code-block:: python

    def make_trainable_vector(d: int) -> Parameter

Creates a trainable parameter vector with initialized embeddings.

**Parameters:**

* **d** *(int)* - Dimension of the vector.

**Returns:**

* **Parameter** - Trainable vector with shape `(d,)`.


**Embedding Modules**
---------------------

class CLSEmbedding(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Appends a learnable [CLS] token to the input tensor (similar to BERT's classification token).

.. code-block:: python

    __init__(self, d_embedding: int) -> None

**Parameters:**

* **d_embedding** *(int)* - Dimension of the embedding.


.. code-block:: python

    forward(self, x: Tensor) -> Tensor

**Parameters:**

* **x** *(Tensor)* - Input tensor with shape `(batch_size, seq_len, d_embedding)`.

**Returns:**

* **Tensor** - Tensor with [CLS] token prepended, shape `(batch_size, seq_len + 1, d_embedding)`.


class LinearEmbeddings(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies linear transformation to each numerical feature to produce embeddings.

.. code-block:: python

    __init__(self, n_features: int, d_embedding: int, bias: bool = True)

**Parameters:**

* **n_features** *(int)* - Number of input features.
* **d_embedding** *(int)* - Dimension of output embeddings.
* **bias** *(bool, optional, Default is True)* - Whether to include a bias term.


.. code-block:: python

    forward(self, x: Tensor) -> Tensor

**Parameters:**

* **x** *(Tensor)* - Input tensor with shape `(batch_size, n_features)`.

**Returns:**

* **Tensor** - Embedded tensor with shape `(batch_size, n_features, d_embedding)`.


class PeriodicEmbeddings(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encodes numerical features using periodic functions (sine/cosine) to capture cyclic patterns.

.. code-block:: python

    __init__(self, n_features: int, n_frequencies: int, frequency_scale: float) -> None

**Parameters:**

* **n_features** *(int)* - Number of input features.
* **n_frequencies** *(int)* - Number of frequency components.
* **frequency_scale** *(float)* - Scale for initializing frequency parameters.


.. code-block:: python

    forward(self, x: Tensor) -> Tensor

**Parameters:**

* **x** *(Tensor)* - Input tensor with shape `(batch_size, n_features)`.

**Returns:**

* **Tensor** - Periodic embeddings with shape `(batch_size, n_features, 2 * n_frequencies)` (sine + cosine components).


class LREmbeddings(nn.Sequential)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear Regression-style embeddings (linear transformation + ReLU activation), from the paper *"On Embeddings for Numerical Features in Tabular Deep Learning"*.

.. code-block:: python

    __init__(self, n_features: int, d_embedding: int) -> None

**Parameters:**

* **n_features** *(int)* - Number of input features.
* **d_embedding** *(int)* - Dimension of output embeddings.


class PLREmbeddings(nn.Sequential)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Periodic Linear Regression embeddings (periodic encoding + linear transformation + ReLU), from the paper *"On Embeddings for Numerical Features in Tabular Deep Learning"*.

.. code-block:: python

    __init__(self, n_features: int, n_frequencies: int, frequency_scale: float, d_embedding: int, lite: bool)

**Parameters:**

* **n_features** *(int)* - Number of input features.
* **n_frequencies** *(int)* - Number of frequency components.
* **frequency_scale** *(float)* - Scale for frequency initialization.
* **d_embedding** *(int)* - Dimension of output embeddings.
* **lite** *(bool)* - If True, uses a shared linear layer; if False, uses feature-specific linear layers.


class PBLDEmbeddings(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Periodic Binned Linear embeddings, extending PLR embeddings with densenet-style feature concatenation.

.. code-block:: python

    __init__(self, n_features: int, n_frequencies: int, frequency_scale: float, d_embedding: int, lite: bool, plr_act_name: str = 'relu', plr_use_densenet: bool = True)

**Parameters:**

* **n_features** *(int)* - Number of input features.
* **n_frequencies** *(int)* - Number of frequency components.
* **frequency_scale** *(float)* - Scale for frequency initialization.
* **d_embedding** *(int)* - Dimension of output embeddings.
* **lite** *(bool)* - Unused in this implementation (retained for compatibility).
* **plr_act_name** *(str, optional, Default is 'relu')* - Activation function ('relu' or 'linear').
* **plr_use_densenet** *(bool, optional, Default is True)* - If True, concatenates original features to embeddings.


.. code-block:: python

    forward(self, x: Tensor) -> Tensor

**Parameters:**

* **x** *(Tensor)* - Input tensor with shape `(batch_size, n_features)`.

**Returns:**

* **Tensor** - Embedded tensor with shape `(batch_size, n_features * d_embedding)` (or with original features concatenated if `plr_use_densenet=True`).


**Network Modules**
-------------------

class ResNet(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~

Residual network for tabular data, with configurable normalization and activation.

.. code-block:: python

    __init__(self, d_in: Optional[int] = None, d_out: Optional[int] = None, n_blocks: int, d_block: int, dropout: float, d_hidden_multiplier: Union[float, int], n_linear_layers_per_block: int = 2, activation: str = 'ReLU', normalization: str, first_normalization: bool)

**Parameters:**

* **d_in** *(Optional[int])* - Input dimension (None for no projection).
* **d_out** *(Optional[int])* - Output dimension (None for no final layer).
* **n_blocks** *(int)* - Number of residual blocks.
* **d_block** *(int)* - Dimension of each block.
* **dropout** *(float)* - Dropout rate.
* **d_hidden_multiplier** *(Union[float, int])* - Multiplier for hidden dimension in 2-layer blocks.
* **n_linear_layers_per_block** *(int, optional, Default is 2)* - Number of linear layers per block (1 or 2).
* **activation** *(str, optional, Default is 'ReLU')* - Activation function name (from `nn`).
* **normalization** *(str)* - Normalization layer name (from `nn` or 'none').
* **first_normalization** *(bool)* - Whether to apply normalization in the first block.


.. code-block:: python

    forward(self, x: Tensor) -> Tensor

**Parameters:**

* **x** *(Tensor)* - Input tensor with shape `(batch_size, d_in)` (or `(batch_size, d_block)` if `d_in` is None).

**Returns:**

* **Tensor** - Output tensor with shape `(batch_size, d_out)` (or `(batch_size, d_block)` if `d_out` is None).


class NLinear(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~

Applies feature-specific linear transformations.

.. code-block:: python

    __init__(self, n_features: int, d_in: int, d_out: int, bias: bool = True)

**Parameters:**

* **n_features** *(int)* - Number of features (each with a separate linear layer).
* **d_in** *(int)* - Input dimension per feature.
* **d_out** *(int)* - Output dimension per feature.
* **bias** *(bool, optional, Default is True)* - Whether to include bias terms.


.. code-block:: python

    forward(self, x: Tensor) -> Tensor

**Parameters:**

* **x** *(Tensor)* - Input tensor with shape `(batch_size, n_features, d_in)`.

**Returns:**

* **Tensor** - Output tensor with shape `(batch_size, n_features, d_out)`.


class MLP(nn.Module)
~~~~~~~~~~~~~~~~~~~~

Multi-layer perceptron for tabular data with sequential blocks.

.. code-block:: python

    __init__(self, d_in: Optional[int] = None, d_out: Optional[int] = None, n_blocks: int, d_block: int, dropout: float, activation: str = 'SELU')

**Parameters:**

* **d_in** *(Optional[int])* - Input dimension (uses `d_block` if None).
* **d_out** *(Optional[int])* - Output dimension (None for no final layer).
* **n_blocks** *(int)* - Number of MLP blocks.
* **d_block** *(int)* - Dimension of each block.
* **dropout** *(float)* - Dropout rate.
* **activation** *(str, optional, Default is 'SELU')* - Activation function name (from `nn`).


.. code-block:: python

    forward(self, x: Tensor) -> Tensor

**Parameters:**

* **x** *(Tensor)* - Input tensor with shape `(batch_size, d_in)` (or `(batch_size, d_block)` if `d_in` is None).

**Returns:**

* **Tensor** - Output tensor with shape `(batch_size, d_out)` (or `(batch_size, d_block)` if `d_out` is None).


**Module Creation Utilities**
-----------------------------

.. code-block:: python

    def make_module(spec, *args, **kwargs) -> nn.Module

Creates a PyTorch module from a specification (string, dict, or callable).

**Parameters:**

* **spec** - Module specification:
  - String: Name of a module in `nn` or custom modules.
  - Dict: Must contain 'type' key with module name, plus other parameters.
  - Callable: Module class or function.
* **args** - Positional arguments passed to the module.
* **kwargs** - Keyword arguments passed to the module.

**Returns:**

* **nn.Module** - Instantiated module.


.. code-block:: python

    def make_module1(type: str, *args, **kwargs) -> nn.Module

Simplified module creation from a type string.

**Parameters:**

* **type** *(str)* - Name of the module (in `nn` or custom modules).
* **args** - Positional arguments passed to the module.
* **kwargs** - Keyword arguments passed to the module.

**Returns:**

* **nn.Module** - Instantiated module.

##References##

.. [Gorishniy2023TabR]
   Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko.
   *TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023*.
   arXiv preprint `2307.14338 <https://arxiv.org/abs/2307.14338>`_, 2023.