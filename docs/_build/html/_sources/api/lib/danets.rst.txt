**DANet**
==========================
Tabular data are ubiquitous in real world applications. Although many commonly-used neural components (e.g., convolution) and extensible neural networks (e.g., ResNet) have been developed by the machine learning community, few of them were effective for tabular data and few designs were adequately tailored for tabular data structures. In this paper, we propose a novel and flexible neural component for tabular data, called Abstract Layer (AbstLay), which learns to explicitly group correlative input features and generate higher-level features for semantics abstraction. Also, we design a structure re-parameterization method to compress AbstLay, thus reducing the computational complexity by a clear margin in the reference phase. A special basic block is built using AbstLays, and we construct a family of Deep Abstract Networks (DANets) for tabular data classification and regression by stacking such blocks. In DANets, a special shortcut path is introduced to fetch information from raw tabular features, assisting feature interactions across different levels. Comprehensive experiments on real-world tabular datasets show that our AbstLay and DANets are effective for tabular data classification and regression, and the computational complexity is superior to competitive methods.


class AcceleratedCreator(object)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A creator class to accelerate neural network modules by processing and modifying layers.

.. code-block:: python

    __init__(self, input_dim, base_out_dim, k)

**Parameters:**

* **input_dim** *(int)* - Input dimension of the network.
* **base_out_dim** *(int)* - Base output dimension for the network layers.
* **k** *(int)* - Parameter used in the Extractor for weight computation.


.. code-block:: python

    __call__(self, network)

Modifies the input network by processing its initial layer and each subsequent layer.

**Parameters:**

* **network** - The neural network to be accelerated.

**Returns:**

* Modified network with processed layers.


.. code-block:: python

    extract_module(self, basicblock, base_input_dim, fix_input_dim)

Extracts and processes individual modules (convolutional and downsample layers) within a basic block.

**Parameters:**

* **basicblock** - The basic block containing layers to process.
* **base_input_dim** *(int)* - Base input dimension for the block.
* **fix_input_dim** *(int)* - Fixed input dimension for downsampling layers.

**Returns:**

* Modified basic block with processed layers.


class Extractor(object)
~~~~~~~~~~~~~~~~~~~~~~~~

A class to extract parameters from abstract layers and compute compressed weights.

.. code-block:: python

    __init__(self, k)

**Parameters:**

* **k** *(int)* - Parameter used for weight computation.


.. code-block:: python

    get_parameter(abs_layer)

Extracts necessary parameters from an abstract layer, including batch normalization stats and weights.

**Parameters:**

* **abs_layer** - The abstract layer to extract parameters from.

**Returns:**

* **alpha** *(torch.Tensor)* - Batch normalization weight data.
* **beta** *(torch.Tensor)* - Batch normalization bias data.
* **eps** *(float)* - Batch normalization epsilon value.
* **mu** *(torch.Tensor)* - Running mean from batch normalization.
* **var** *(torch.Tensor)* - Running variance from batch normalization.
* **sparse_weight** *(torch.Tensor)* - Sparse weights from the masker.
* **process_weight** *(torch.Tensor)* - Weight data from the fully connected layer.
* **process_bias** *(torch.Tensor or None)* - Bias data from the fully connected layer (if exists).


.. code-block:: python

    compute_weights(a, b, eps, mu, var, sw, pw, pb, base_input_dim, base_output_dim, k)

Computes compressed weights and biases using extracted parameters.

**Parameters:**

* **a** *(torch.Tensor)* - Alpha (batch norm weight).
* **b** *(torch.Tensor)* - Beta (batch norm bias).
* **eps** *(float)* - Batch norm epsilon.
* **mu** *(torch.Tensor)* - Running mean.
* **var** *(torch.Tensor)* - Running variance.
* **sw** *(torch.Tensor)* - Sparse weight.
* **pw** *(torch.Tensor)* - Process weight.
* **pb** *(torch.Tensor or None)* - Process bias.
* **base_input_dim** *(int)* - Base input dimension.
* **base_output_dim** *(int)* - Base output dimension.
* **k** *(int)* - Parameter for weight shaping.

**Returns:**

* **W_att** *(torch.Tensor)* - Attention weights.
* **W_fc** *(torch.Tensor)* - Feature weights.
* **B_att** *(torch.Tensor)* - Attention biases.
* **B_fc** *(torch.Tensor)* - Feature biases.


.. code-block:: python

    __call__(self, abslayer, input_dim, base_out_dim)

Processes an abstract layer to create a compressed layer.

**Parameters:**

* **abslayer** - The abstract layer to process.
* **input_dim** *(int)* - Input dimension of the layer.
* **base_out_dim** *(int)* - Base output dimension of the layer.

**Returns:**

* **CompressAbstractLayer** - Instance with computed weights and biases.


class CompressAbstractLayer(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A compressed abstract layer module for efficient forward computation.

.. code-block:: python

    __init__(self, att_w, f_w, att_b, f_b)

**Parameters:**

* **att_w** *(torch.Tensor)* - Attention weights.
* **f_w** *(torch.Tensor)* - Feature weights.
* **att_b** *(torch.Tensor)* - Attention biases.
* **f_b** *(torch.Tensor)* - Feature biases.


.. code-block:: python

    forward(self, x)

Performs forward pass using attention and feature weights.

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor of shape [batch_size, input_dim].

**Returns:**

* **torch.Tensor** - Output tensor after processing, shape [batch_size, output_dim].


**sparsemax.py Components**
---------------------------

.. code-block:: python

    _make_ix_like(input, dim=0)

Creates an index tensor with a similar shape to the input tensor.

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor to match shape with.
* **dim** *(int, optional, Default is 0)* - Dimension to align the index tensor with.

**Returns:**

* **torch.Tensor** - Index tensor with shape matching the input.


class SparsemaxFunction(Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autograd function implementing the sparsemax activation (Martins & Astudillo, 2016).

.. code-block:: python

    forward(ctx, input, dim=-1)

Forward pass to compute sparsemax activation.

**Parameters:**

* **ctx** - Context object to store variables for backward pass.
* **input** *(torch.Tensor)* - Input tensor of any shape.
* **dim** *(int, optional, Default is -1)* - Dimension along which to apply sparsemax.

**Returns:**

* **torch.Tensor** - Output tensor with the same shape as input, containing sparse probabilities.


.. code-block:: python

    backward(ctx, grad_output)

Backward pass to compute gradients of the loss with respect to the input.

**Parameters:**

* **ctx** - Context object with stored variables from forward pass.
* **grad_output** *(torch.Tensor)* - Gradient of the loss with respect to the output.

**Returns:**

* **torch.Tensor** - Gradient of the loss with respect to the input.
* **None** - No gradient for the `dim` parameter.


.. code-block:: python

    _threshold_and_support(input, dim=-1)

Computes the threshold and support size for sparsemax.

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor.
* **dim** *(int, optional, Default is -1)* - Dimension to compute over.

**Returns:**

* **tau** *(torch.Tensor)* - Threshold value for sparsemax.
* **support_size** *(torch.Tensor)* - Number of non-zero elements (support size).


class Sparsemax(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Module wrapping the sparsemax activation function.

.. code-block:: python

    __init__(self, dim=-1)

**Parameters:**

* **dim** *(int, optional, Default is -1)* - Dimension along which to apply sparsemax.


.. code-block:: python

    forward(self, input)

Applies sparsemax activation to the input.

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor.

**Returns:**

* **torch.Tensor** - Output after sparsemax activation.


class Entmax15Function(Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autograd function implementing Entmax with alpha=1.5 (Peters et al., 2019).

.. code-block:: python

    forward(ctx, input, dim=-1)

Forward pass to compute Entmax15 activation.

**Parameters:**

* **ctx** - Context object to store variables for backward pass.
* **input** *(torch.Tensor)* - Input tensor of any shape.
* **dim** *(int, optional, Default is -1)* - Dimension along which to apply Entmax15.

**Returns:**

* **torch.Tensor** - Output tensor with the same shape as input.


.. code-block:: python

    backward(ctx, grad_output)

Backward pass to compute gradients of the loss with respect to the input.

**Parameters:**

* **ctx** - Context object with stored variables from forward pass.
* **grad_output** *(torch.Tensor)* - Gradient of the loss with respect to the output.

**Returns:**

* **torch.Tensor** - Gradient of the loss with respect to the input.
* **None** - No gradient for the `dim` parameter.


.. code-block:: python

    _threshold_and_support(input, dim=-1)

Computes the threshold and support size for Entmax15.

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor.
* **dim** *(int, optional, Default is -1)* - Dimension to compute over.

**Returns:**

* **tau_star** *(torch.Tensor)* - Threshold value for Entmax15.
* **support_size** *(torch.Tensor)* - Number of non-zero elements (support size).


class Entmoid15(Function)
~~~~~~~~~~~~~~~~~~~~~~~~~

Optimized autograd function equivalent to Entmax15([x, 0]).

.. code-block:: python

    forward(ctx, input)

Forward pass to compute the Entmoid15 activation.

**Parameters:**

* **ctx** - Context object to store output for backward pass.
* **input** *(torch.Tensor)* - Input tensor.

**Returns:**

* **torch.Tensor** - Output tensor with the same shape as input.


.. code-block:: python

    backward(ctx, grad_output)

Backward pass to compute gradients.

**Parameters:**

* **ctx** - Context object with stored output from forward pass.
* **grad_output** *(torch.Tensor)* - Gradient of the loss with respect to the output.

**Returns:**

* **torch.Tensor** - Gradient of the loss with respect to the input.


class Sparsemax(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Module wrapper for the sparsemax activation function.

.. code-block:: python

    __init__(self, dim=-1)

**Parameters:**

* **dim** *(int, optional, Default is -1)* - Dimension along which to apply sparsemax.


.. code-block:: python

    forward(self, input)

Applies sparsemax activation to the input.

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor.

**Returns:**

* **torch.Tensor** - Output after sparsemax activation.


class Entmax15(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~

Module wrapper for the Entmax15 activation function.

.. code-block:: python

    __init__(self, dim=-1)

**Parameters:**

* **dim** *(int, optional, Default is -1)* - Dimension along which to apply Entmax15.


.. code-block:: python

    forward(self, input)

Applies Entmax15 activation to the input.

**Parameters:**

* **input** *(torch.Tensor)* - Input tensor.

**Returns:**

* **torch.Tensor** - Output after Entmax15 activation.

**Referencses:**

Chen, J., Liao, K., Wan, Y., Chen, D. Z., & Wu, J. (2022). DANets: Deep Abstract Networks for Tabular Data Classification and Regression. arXiv:2112.02962 [cs.LG]. `<https://arxiv.org/abs/2112.02962>`_