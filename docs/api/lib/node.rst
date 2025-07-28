**NODE**
=========================================
A tree-mimic method that generalizes oblivious decision trees, combining gradient-based optimization with hierarchical representation learning.

class DenseBlock(nn.Sequential)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A dense block composed of multiple layers of tree-based modules (e.g., ODST) with residual connections. Each layer concatenates its output with the input features, creating a densely connected structure.

.. code-block:: python

    __init__(self, input_dim, layer_dim, num_layers, tree_dim=1, max_features=None, input_dropout=0.0, flatten_output=True, Module=ODST, **kwargs)

**Parameters:**

* **input_dim** *(int)* - Number of input features.
* **layer_dim** *(int)* - Number of trees (output features) per layer.
* **num_layers** *(int)* - Number of layers in the dense block.
* **tree_dim** *(int, optional, Default is 1)* - Number of outputs per tree.
* **max_features** *(Optional[int], Default is None)* - Maximum number of features to retain (truncates concatenated features if exceeded).
* **input_dropout** *(float, optional, Default is 0.0)* - Dropout rate applied to layer inputs during training.
* **flatten_output** *(bool, optional, Default is True)* - Whether to flatten the output or keep tree outputs separate.
* **Module** *(type, optional, Default is ODST)* - Module class to use for each layer (e.g., ODST).
* **kwargs** - Additional arguments passed to the module class.


**Forward Pass**
----------------

.. code-block:: python

    forward(self, x)

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor with shape `(batch_size, ..., input_dim)`.

**Returns:**

* **torch.Tensor** - Output tensor with shape:
  - If `flatten_output=True`: `(batch_size, ..., num_layers * layer_dim * tree_dim)`.
  - If `flatten_output=False`: `(batch_size, ..., num_layers * layer_dim, tree_dim)`.


.. code-block:: python

    to_one_hot(y, depth=None)

Converts integer tensor to one-hot encoding.

**Parameters:**

* **y** *(torch.Tensor)* - Input integer tensor of any shape.
* **depth** *(Optional[int], Default is None)* - Size of one-hot dimension. If None, inferred as `max(y) + 1`.

**Returns:**

* **torch.Tensor** - One-hot encoded tensor with shape `(*y.shape, depth)`.




class SparsemaxFunction(Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies the Sparsemax function, a sparse alternative to softmax.

.. code-block:: python

    forward(ctx, input, dim=-1)

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor of any shape.
* **dim** *(int, optional, Default is -1)* - Dimension along which to apply Sparsemax.

**Returns:**

* **torch.Tensor** - Output tensor with same shape as input, values summing to 1 along `dim`.


.. code-block:: python

    backward(ctx, grad_output)

**Parameters:**

* **grad_output** *(torch.Tensor)* - Gradient of the loss with respect to the output.

**Returns:**

* **torch.Tensor** - Gradient of the loss with respect to the input.


class Entmax15Function(Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies the Entmax 1.5 function, a sparse and smooth alternative to softmax.

.. code-block:: python

    forward(ctx, input, dim=-1)

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor of any shape.
* **dim** *(int, optional, Default is -1)* - Dimension along which to apply Entmax15.

**Returns:**

* **torch.Tensor** - Output tensor with same shape as input, values summing to 1 along `dim`.


.. code-block:: python

    backward(ctx, grad_output)

**Parameters:**

* **grad_output** *(torch.Tensor)* - Gradient of the loss with respect to the output.

**Returns:**

* **torch.Tensor** - Gradient of the loss with respect to the input.


**Entmoid15 Activation**
------------------------

class Entmoid15(Function)
~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient implementation of Entmax15 for binary classification.

.. code-block:: python

    forward(ctx, input)

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor of any shape.

**Returns:**

* **torch.Tensor** - Output tensor with values between 0 and 1.


.. code-block:: python

    backward(ctx, grad_output)

**Parameters:**

* **grad_output** *(torch.Tensor)* - Gradient of the loss with respect to the output.

**Returns:**

* **torch.Tensor** - Gradient of the loss with respect to the input.


**Utility Modules**
-------------------

class Lambda(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~

Wraps a function into a PyTorch module.

.. code-block:: python

    __init__(self, func)

**Parameters:**

* **func** *(callable)* - Function to wrap.


.. code-block:: python

    forward(self, *args, **kwargs)

**Parameters:**

* **args** - Positional arguments passed to the function.
* **kwargs** - Keyword arguments passed to the function.

**Returns:**

* **Any** - Output of the function.


class ModuleWithInit(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base class for modules with data-aware initialization on the first batch.

.. code-block:: python

    initialize(self, *args, **kwargs)

**Parameters:**

* **args** - Positional arguments for initialization.
* **kwargs** - Keyword arguments for initialization.

**Note:**
Must be implemented by subclasses to initialize module parameters using the first batch of data.


.. code-block:: python

    __call__(self, *args, **kwargs)

**Parameters:**

* **args** - Positional arguments for the forward pass.
* **kwargs** - Keyword arguments for the forward pass.

**Returns:**

* **torch.Tensor** - Output of the module's forward pass.


class ODST(ModuleWithInit)
~~~~~~~~~~~~~~~~~~~~~~~~~~

A differentiable tree-based module that combines oblivious decision trees with sparse activation functions (sparsemax/sparsemoid) for end-to-end training. Designed as a drop-in replacement for `nn.Linear` with enhanced feature interaction capabilities.

.. code-block:: python

    __init__(self, in_features, num_trees, depth=6, tree_dim=1, flatten_output=True, choice_function=sparsemax, bin_function=sparsemoid, initialize_response_=nn.init.normal_, initialize_selection_logits_=nn.init.uniform_, threshold_init_beta=1.0, threshold_init_cutoff=1.0)

**Parameters:**

* **in_features** *(int)* - Number of input features.
* **num_trees** *(int)* - Number of decision trees in the module.
* **depth** *(int, optional, Default is 6)* - Depth of each tree (number of splits).
* **tree_dim** *(int, optional, Default is 1)* - Number of output channels per tree.
* **flatten_output** *(bool, optional, Default is True)* - If True, flattens output to `(batch_size, num_trees * tree_dim)`; otherwise, returns `(batch_size, num_trees, tree_dim)`.
* **choice_function** *(Callable, optional, Default is sparsemax)* - Function to compute feature weights (must map to a simplex, e.g., sparsemax).
* **bin_function** *(Callable, optional, Default is sparsemoid)* - Function to compute leaf weights (must map to [0, 1], e.g., sparsemoid).
* **initialize_response_** *(Callable, optional, Default is nn.init.normal_)* - Initializer for tree output responses.
* **initialize_selection_logits_** *(Callable, optional, Default is nn.init.uniform_)* - Initializer for feature selection logits.
* **threshold_init_beta** *(float, optional, Default is 1.0)* - Beta distribution parameter for data-aware threshold initialization (controls quantile selection).
* **threshold_init_cutoff** *(float, optional, Default is 1.0)* - Scaling factor for temperature initialization (controls sparsity of bin selections).


**Forward Pass**
----------------

.. code-block:: python

    forward(self, input)

Processes input through oblivious decision trees with sparse activations.

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor with shape `(batch_size, ..., in_features)`.

**Returns:**

* **torch.Tensor** - Output tensor with shape:
  - If `flatten_output=True`: `(batch_size, ..., num_trees * tree_dim)`.
  - If `flatten_output=False`: `(batch_size, ..., num_trees, tree_dim)`.


**Data-Aware Initialization**
-----------------------------

.. code-block:: python

    initialize(self, input, eps=1e-6)

Initializes tree thresholds and temperatures based on input data (called automatically on first forward pass).

**Parameters:**
* **input** *(torch.Tensor)* - Input data tensor with shape `(batch_size, in_features)` (used for initialization).
* **eps** *(float, optional, Default is 1e-6)* - Small epsilon to avoid division by zero.

**Initialization Logic:**
1. **Thresholds**: Sampled from data quantiles using a Beta distribution (controlled by `threshold_init_beta`).
2. **Temperatures**: Scaled to ensure most data points fall in the linear region of `bin_function` (controlled by `threshold_init_cutoff`).


**Repr**
--------

.. code-block:: python

    __repr__(self)

Returns a string representation of the ODST module with key parameters.

**References:**

Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data
Sergei Popov, Stanislav Morozov, Artem Babenko
arXiv preprint arXiv:1909.06312, 2019.

PDF `<https://arxiv.org/abs/1909.06312>`_
