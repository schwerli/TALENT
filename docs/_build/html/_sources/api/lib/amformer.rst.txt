**amformer**
==============

A modified transformer architecture enabling arithmetical feature interactions


Class GEGLU (nn.Module)
---------------------------

Functions
~~~~~~~~~

.. code-block:: python

    forward(self, x)

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor

**Returns:**

* **torch.Tensor** - Output tensor after applying GEGLU activation


.. code-block:: python

    def FeedForward(dim, mult = 4, dropout = 0.)

**Parameters:**

* **dim** *(int)* - Input dimension
* **mult** *(int, optional, Default is 4)* - Expansion factor for the hidden layer size relative to `dim`
* **dropout** *(float, optional, Default is 0)* - Dropout probability applied between the two linear layers

**Returns:**

* **FeedForward** *(nn.Module)* - A feed-forward block that applies two linear transformations with an activation and optional dropout in between


class Attention(nn.Module)
---------------------------

.. code-block:: python

    __init__(self, heads = 8, dim = 64, dropout = 0., inner_dim = 0)

**Parameters:**

* **heads** *(int, optional, Default is 8)* - Number of attention heads
* **dim** *(int, optional, Default is 64)* - Size of the input embedding
* **dropout** *(float, optional, Default is 0)* - Dropout probability applied to the attention weights
* **inner_dim** *(int, optional, Default is 0)* - Size of the inner dimension for the attention block. If set to 0, it will be set to `dim`


.. code-block:: python

    forward(self, x, attn_out = False)

Computes the forward pass of the attention mechanism using scaled dot-product attention.

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor of shape [batch_size, seq_len, dim]
* **attn_out** *(bool, optional, Default is False)* - If True, returns both the output tensor and the attention weights

**Returns:**

* **torch.Tensor** - Output tensor after applying attention mechanism of shape [batch_size, seq_len, dim]
* **torch.Tensor** - Attention weights (only returned if `attn_out` is True) of shape [batch_size, heads, seq_len, seq_len]



class MemoryBlock(nn.Module)
------------------------------

Functions
~~~~~~~~~

.. code-block:: python

    __init__(self, token_num, heads, dim, attn_dropout, cluster, target_mode, groups, num_per_group, use_cls_token, sum_or_prod = None, qk_relu = False)

**Parameters:**

* **token_num** *(int)* - Number of tokens
* **heads** *(int)* - Number of attention heads
* **dim** *(int)* - Dimension of the input embedding
* **attn_dropout** *(float)* - Dropout probability for attention weights
* **cluster** *(bool)* - Whether to use clustering for target tokens
* **target_mode** *(str)* - Mode for target token generation ('mix' or other)
* **groups** *(int)* - Number of groups for token processing
* **num_per_group** *(int)* - Number of tokens to gather per group. If -1, no grouping is applied
* **use_cls_token** *(bool)* - Whether to use a classification token
* **sum_or_prod** *(str, optional, Default is None)* - Specifies if the block performs 'sum' or 'prod' aggregation. Must be 'sum' or 'prod'
* **qk_relu** *(bool, optional, Default is False)* - Whether to apply ReLU activation to query (Q) and key (K) tensors

.. code-block:: python

    forward(self, x)

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor

**Returns:**

* **torch.Tensor** - Output tensor after applying MemoryBlock operations


class Transformer(nn.Module)
-------------------------------

Functions
~~~~~~~~~

.. code-block:: python

    __init__(self, dim, depth, heads, attn_dropout, ff_dropout, use_cls_token, groups, sum_num_per_group, prod_num_per_group, cluster, target_mode, token_num, token_descent=False, use_prod=True, qk_relu = False)

**Parameters:**

* **dim** *(int)* - Dimension of the input embedding
* **depth** *(int)* - Number of transformer layers
* **heads** *(int)* - Number of attention heads
* **attn_dropout** *(float)* - Dropout probability for attention weights in MemoryBlocks
* **ff_dropout** *(float)* - Dropout probability for feed-forward layers
* **use_cls_token** *(bool)* - Whether to use a classification token in MemoryBlocks
* **groups** *(list of int)* - List of group numbers for each layer
* **sum_num_per_group** *(list of int)* - List of `num_per_group` for 'sum' MemoryBlocks in each layer
* **prod_num_per_group** *(list of int)* - List of `num_per_group` for 'prod' MemoryBlocks in each layer
* **cluster** *(bool)* - Whether to use clustering in MemoryBlocks
* **target_mode** *(str)* - Mode for target token generation in MemoryBlocks
* **token_num** *(int)* - Initial number of tokens
* **token_descent** *(bool, optional, Default is False)* - If True, the number of tokens can decrease across layers
* **use_prod** *(bool, optional, Default is True)* - Whether to include product-based MemoryBlocks in the transformer layers
* **qk_relu** *(bool, optional, Default is False)* - Whether to apply ReLU activation to query (Q) and key (K) tensors in MemoryBlocks

.. code-block:: python

    forward(self, x)

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor

**Returns:**

* **torch.Tensor** - Output tensor after applying transformer layers


class NumericalEmbedder(nn.Module)
------------------------------------

Functions
~~~~~~~~~

.. code-block:: python

    __init__(self, dim, num_numerical_types)

**Parameters:**

* **dim** *(int)* - Dimension of the output embedding
* **num_numerical_types** *(int)* - Number of different numerical features to embed

.. code-block:: python

    forward(self, x)

**Parameters:**

* **x** *(torch.Tensor)* - Input numerical tensor

**Returns:**

* **torch.Tensor** - Embedded numerical features

**Referencses:**

Cheng, Y., Hu, R., Ying, H., Shi, X., Wu, J., & Lin, W. (2024). Arithmetic Feature Interaction Is Necessary for Deep Tabular Learning. arXiv:2402.02334. `<https://arxiv.org/abs/2402.02334>`_
