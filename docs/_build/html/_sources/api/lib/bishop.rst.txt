**BiSHop**
=============================================

BiSHop leverages a sparse Hopfield model with adaptable sparsity, enhanced by column-wise and row-wise modules. It's specifically designed to address challenges in processing rotationally invariant and sparse tabular data.


class GSH(torch.nn.Module)
---------------------------

**Generalized Sparse Hopfield module**

A generic sparse Hopfield module that implements an attention mechanism based on generalized sparse activation functions.

.. code-block:: python

    __init__(self, scale=None, dropout=0.1, actv='sparsemax')

**Parameters:**

* **scale** *(float, optional, Default is None)* - Scaling factor for attention scores. If None, it is automatically set to 1/√(feature dimension).
* **dropout** *(float, optional, Default is 0.1)* - Dropout ratio applied to attention weights.
* **actv** *(str, optional, Default is 'sparsemax')* - Type of activation function. Options:
  - 'softmax': Uses standard Softmax.
  - 'sparsemax': Uses Sparsemax sparse activation.
  - Other: Uses EntmaxAlpha activation.


.. code-block:: python

    forward(self, queries, keys, values)

**Parameters:**

* **queries** *(torch.Tensor)* - Query tensor with shape [batch_size, query_seq_len, num_heads, head_dim].
* **keys** *(torch.Tensor)* - Key tensor with shape [batch_size, key_seq_len, num_heads, head_dim].
* **values** *(torch.Tensor)* - Value tensor with shape [batch_size, value_seq_len, num_heads, value_dim].

**Returns:**

* **torch.Tensor** - Attention output tensor with shape [batch_size, query_seq_len, num_heads, value_dim], in contiguous memory layout.


class GSHLayer(torch.nn.Module)
-----------------------------------

**Generalized Sparse Hopfield (GSH) layer**

A generalized sparse Hopfield attention layer that encapsulates the complete implementation of multi-head attention mechanism.

.. code-block:: python

    __init__(self, d_model=512, n_heads=8, d_keys=None, d_values=None, mix=True, dropout=0.1, actv='entmax', hopfield=True)


**Parameters:**

* **d_model** *(int, optional, Default is 512)* - Hidden feature dimension.
* **n_heads** *(int, optional, Default is 8)* - Number of attention heads.
* **d_keys** *(int, optional, Default is None)* - Projection dimension for queries and keys. If None, set to d_model//n_heads.
* **d_values** *(int, optional, Default is None)* - Projection dimension for values. If None, set to d_model//n_heads.
* **mix** *(bool, optional, Default is True)* - Whether to mix the dimension order of the output in the forward pass.
* **dropout** *(float, optional, Default is 0.1)* - Dropout ratio.
* **actv** *(str, optional, Default is 'entmax')* - Type of activation function (same as GSH module).
* **hopfield** *(bool, optional, Default is True)* - Whether to use the Hopfield attention mechanism. If False, uses classical Transformer attention.


.. code-block:: python

    forward(self, queries, keys, values)

**Parameters:**

* **queries** *(torch.Tensor)* - Query input tensor with shape [batch_size, query_seq_len, d_model].
* **keys** *(torch.Tensor)* - Key input tensor with shape [batch_size, key_seq_len, d_model].
* **values** *(torch.Tensor)* - Value input tensor with shape [batch_size, value_seq_len, d_model].

**Returns:**

* **torch.Tensor** - Attention layer output tensor with shape [batch_size, query_seq_len, d_model].


class BAModule(torch.nn.Module)
-------------------------------

**BAModule (Feature and Embedding Interaction Module)**

A feature and embedding interaction module that combines cross-feature attention and cross-embedding attention to achieve complex feature interactions.

.. code-block:: python

    __init__(self, n_pool=10, factor=8, d_model=512, n_heads=8, d_ff=None, dropout=0.1, actv='entmax', hopfield=True)

**Parameters:**

* **n_pool** *(int, optional, Default is 10)* - Number of pooling vectors.
* **factor** *(int, optional, Default is 8)* - Factor dimension for pooling vectors.
* **d_model** *(int, optional, Default is 512)* - Hidden feature dimension.
* **n_heads** *(int, optional, Default is 8)* - Number of attention heads.
* **d_ff** *(int, optional, Default is None)* - Hidden layer dimension for the feedforward network. If None, set to 4*d_model.
* **dropout** *(float, optional, Default is 0.1)* - Dropout ratio.
* **actv** *(str, optional, Default is 'entmax')* - Type of activation function (same as GSH module).
* **hopfield** *(bool, optional, Default is True)* - Whether to use the Hopfield attention mechanism.


.. code-block:: python

    forward(self, x)

**Parameters:**

* **x** *(torch.Tensor)* - Input patched embedded tabular data with shape [batch_size, embedding_dim, num_patches, d_model].

**Returns:**

* **torch.Tensor** - Processed patched embedded tabular data with shape [batch_size, embedding_dim, n_pool, d_model].



class DecoderLayer(torch.nn.Module)
------------------------------------

A single decoder layer combining BAModule and GSHLayer for cross-attention and feature refinement.

.. code-block:: python

    __init__(self, patch_dim=10, n_pool=10, factor=10, actv='entmax', hopfield=True, d_model=512, n_heads=8, d_ff=1024, dropout=0.2)

**Parameters:**

* **patch_dim** *(int, optional, Default is 10)* - Dimension of output patches (segment length).
* **n_pool** *(int, optional, Default is 10)* - Number of pooling vectors for BAModule.
* **factor** *(int, optional, Default is 10)* - Factor dimension for pooling vectors in BAModule.
* **actv** *(str, optional, Default is 'entmax')* - Activation function type for attention mechanisms.
* **hopfield** *(bool, optional, Default is True)* - Whether to use Hopfield attention (True) or classical Transformer attention (False).
* **d_model** *(int, optional, Default is 512)* - Hidden feature dimension.
* **n_heads** *(int, optional, Default is 8)* - Number of attention heads.
* **d_ff** *(int, optional, Default is 1024)* - Hidden layer dimension for feedforward networks in BAModule.
* **dropout** *(float, optional, Default is 0.2)* - Dropout probability applied throughout the layer.


.. code-block:: python

    forward(self, x, enc_x)

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor to the decoder layer, shape: [batch_size, embedding_dim, num_patches, d_model].
* **enc_x** *(torch.Tensor)* - Encoded input tensor from the encoder, shape: [batch_size, embedding_dim, n_pool, d_model].

**Returns:**

* **dec_out** *(torch.Tensor)* - Refined decoder output before final projection, shape: [batch_size, embedding_dim, n_out, d_model].
* **layer_predict** *(torch.Tensor)* - Layer-specific prediction tensor, shape: [batch_size, (embedding_dim * n_out), patch_dim].

**Description:**
1. Processes input through BAModule to enhance feature interactions.
2. Applies cross-attention (via GSHLayer) between decoder input and encoder output.
3. Refines features with residual connections, layer normalization, and a small MLP.
4. Projects outputs to the target patch dimension for prediction.


class Decoder(torch.nn.Module)
------------------------------

A stacked decoder consisting of multiple DecoderLayer instances for iterative refinement.

.. code-block:: python

    __init__(self, d_layer=3, patch_dim=10, n_pool=10, factor=10, actv='entmax', hopfield=True, d_model=512, n_heads=8, d_ff=1024, dropout=0.2)

**Parameters:**

* **d_layer** *(int, optional, Default is 3)* - Number of stacked DecoderLayer instances.
* **patch_dim** *(int, optional, Default is 10)* - Dimension of output patches (passed to DecoderLayer).
* **n_pool** *(int, optional, Default is 10)* - Number of pooling vectors (passed to DecoderLayer).
* **factor** *(int, optional, Default is 10)* - Factor dimension for pooling vectors (passed to DecoderLayer).
* **actv** *(str, optional, Default is 'entmax')* - Activation function type for attention mechanisms.
* **hopfield** *(bool, optional, Default is True)* - Whether to use Hopfield attention in submodules.
* **d_model** *(int, optional, Default is 512)* - Hidden feature dimension.
* **n_heads** *(int, optional, Default is 8)* - Number of attention heads.
* **d_ff** *(int, optional, Default is 1024)* - Feedforward network dimension (passed to DecoderLayer).
* **dropout** *(float, optional, Default is 0.2)* - Dropout probability applied in all submodules.


.. code-block:: python

    forward(self, x, enc)

**Parameters:**

* **x** *(torch.Tensor)* - Initial decoder input tensor, shape: [batch_size, embedding_dim, num_patches, d_model].
* **enc** *(list of torch.Tensor)* - List of encoder outputs, with length equal to `d_layer`. Each tensor has shape: [batch_size, embedding_dim, n_pool, d_model].

**Returns:**

* **final_predict** *(torch.Tensor)* - Aggregated final prediction, shape: [batch_size, (n_patch * patch_dim), embedding_dim].


class NumEmb(torch.nn.Module)
------------------------------

**Numerical embedding for tabular data**

Converts numerical features into continuous embeddings using quantile-based binning and linear interpolation.

.. code-block:: python

    __init__(self, n_num, emb_dim)

**Parameters:**

* **n_num** *(int)* - Number of numerical features.
* **emb_dim** *(int)* - Dimension of the output embedding for each numerical feature. Must be greater than 0.


.. code-block:: python

    get_bins(self, data, identifier='num')

Computes quantile bins for numerical features based on input data.

**Parameters:**

* **data** *(torch.utils.data.DataLoader or torch.Tensor)* - Input data containing numerical features. If DataLoader, batches are concatenated.
* **identifier** *(str, optional, Default is 'num')* - Key for numerical features if `data` is a DataLoader of dictionaries.

**Description:**
Calculates quantiles for each numerical feature using linearly spaced bins (from 0 to 1). These quantiles are stored in `self.quantiles` for use in the forward pass.


.. code-block:: python

    forward(self, x)

Generates embeddings for numerical features using precomputed quantile bins.

**Parameters:**

* **x** *(torch.Tensor)* - Input numerical tensor of shape [batch_size, n_num].

**Returns:**

* **torch.Tensor** - Embedded numerical features of shape [batch_size, n_num, emb_dim], with values in [0, 1].

**Description:**
Maps each numerical value to a continuous embedding by:
- Checking its position relative to precomputed quantile bins.
- Assigning 0 if below the lower bin, 1 if above the upper bin, and linear interpolation between bins otherwise.


.. code-block:: python

    _to(self, device)

Moves internal bins and quantiles to the specified device.

**Parameters:**

* **device** - Target device (e.g., 'cpu' or 'cuda').


class FullEmbDropout(torch.nn.Module)
--------------------------------------

Applies dropout to entire embedding features (full feature dropout).

.. code-block:: python

    __init__(self, dropout: float=0.1)

**Parameters:**

* **dropout** *(float, optional, Default is 0.1)* - Probability of dropping an entire feature.


.. code-block:: python

    forward(self, X: torch.Tensor) -> torch.Tensor

Applies full feature dropout to the input tensor.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor of shape [batch_size, num_features, emb_dim].

**Returns:**

* **torch.Tensor** - Tensor with full feature dropout applied, same shape as input.

**Description:**
Generates a binary mask (per feature) to drop entire features with probability `dropout`, scaled by 1/(1 - dropout) to maintain mean.


class _Embedding(torch.nn.Embedding)
-------------------------------------

Custom embedding layer with truncated normal initialization.

**Description:**
Inherits from `torch.nn.Embedding` but initializes weights using a truncated normal distribution (approximation) for better stability.

.. code-block:: python

    __init__(self, ni, nf, std=0.01)

**Parameters:**
* **ni** *(int)* - Number of input classes (vocabulary size).
* **nf** *(int)* - Size of each embedding vector.
* **std** *(float, optional, Default is 0.01)* - Standard deviation for truncated normal initialization.


class SharedEmbedding(torch.nn.Module)
---------------------------------------

Embedding layer with optional shared components across all classes.

.. code-block:: python

    __init__(self, n_class, emb_dim, share=True, share_add=False, share_div=8)

**Parameters:**

* **n_class** *(int)* - Number of classes for embedding.
* **emb_dim** *(int)* - Total dimension of the output embedding.
* **share** *(bool, optional, Default is True)* - Whether to include a shared embedding component.
* **share_add** *(bool, optional, Default is False)* - If True, adds the shared component to class-specific embeddings; if False, concatenates them.
* **share_div** *(int, optional, Default is 8)* - Factor to determine the size of the shared component (only used if `share_add` is False).


.. code-block:: python

    forward(self, x)

Generates embeddings with optional shared components.

**Parameters:**

* **x** *(torch.Tensor)* - Input class indices of shape [batch_size].

**Returns:**

* **torch.Tensor** - Embedded tensor of shape [batch_size, 1, emb_dim].

**Description:**
- If `share` is True, combines class-specific embeddings with a learnable shared component (either via addition or concatenation).
- If `share` is False, behaves like a standard embedding layer.


class CatEmb(torch.nn.Module)
------------------------------

**Categorical embedding for tabular data**

Handles embedding for multiple categorical features, with options for shared components and dropout.

.. code-block:: python

    __init__(self, n_cat, emb_dim, n_class, share=True, share_add=False, share_div=8, full_dropout=False, emb_dropout=0.1)

**Parameters:**

* **n_cat** *(int)* - Number of categorical features.
* **emb_dim** *(int)* - Dimension of the output embedding for each categorical feature.
* **n_class** *(int)* - Number of classes for each categorical feature.
* **share** *(bool, optional, Default is True)* - Whether to use `SharedEmbedding` for each feature.
* **share_add** *(bool, optional, Default is False)* - Passed to `SharedEmbedding` (add vs. concatenate shared component).
* **share_div** *(int, optional, Default is 8)* - Passed to `SharedEmbedding` (determines shared component size).
* **full_dropout** *(bool, optional, Default is False)* - If True, uses `FullEmbDropout`; otherwise, standard dropout.
* **emb_dropout** *(float, optional, Default is 0.1)* - Dropout probability for embeddings.


.. code-block:: python

    forward(self, x)

Generates embeddings for multiple categorical features.

**Parameters:**

* **x** *(torch.Tensor)* - Input categorical indices of shape [batch_size, n_cat].

**Returns:**

* **torch.Tensor** - Embedded categorical features of shape [batch_size, n_cat, emb_dim].

**Description:**
- Applies embedding layers (either `SharedEmbedding` or standard `Embedding`) to each categorical feature.
- Concatenates embeddings across features and applies dropout (full feature dropout or standard dropout).


class PatchEmb(torch.nn.Module)
-------------------------------

**Patch embedding for aggregating features**

Splits embedded features into patches and projects them to a target dimension for attention mechanisms.

.. code-block:: python

    __init__(self, patch_dim, d_model)

**Parameters:**

* **patch_dim** *(int)* - Number of features per patch.
* **d_model** *(int)* - Dimension of the projected patch embeddings (matches the attention mechanism's input dimension).


.. code-block:: python

    forward(self, x)

Converts embedded features into patched embeddings.

**Parameters:**

* **x** *(torch.Tensor)* - Input embedded tabular data of shape [batch_size, feature_dim, embedding_dim].

**Returns:**

* **torch.Tensor** - Patched embedded data of shape [batch_size, embedding_dim, num_patches, d_model], where `num_patches = feature_dim // patch_dim`.

**Description:**
1. Splits the input features into non-overlapping patches, each containing `patch_dim` features.
2. Projects each patch to `d_model` dimension using a linear layer.
3. Rearranges the output to align with the expected input shape for attention mechanisms.


class PatchMerge(torch.nn.Module)
----------------------------------

**Merge adjacent patches together**

Aggregates multiple adjacent patches into a single patch using linear projection.

.. code-block:: python

    __init__(self, d_model=512, n_agg=4)

**Parameters:**

* **d_model** *(int, optional, Default is 512)* - Number of features in each patch.
* **n_agg** *(int, optional, Default is 4)* - Number of adjacent patches to aggregate.


.. code-block:: python

    forward(self, x)

Merges adjacent patches into aggregated patches.

**Parameters:**

* **x** *(torch.Tensor)* - Input patched tensor of shape [batch_size, emb_dim, n_patch, d_model]

**Returns:**

* **torch.Tensor** - Merged patched tensor of shape [batch_size, emb_dim, merged_n_patch, d_model], where merged_n_patch = ceil(n_patch / n_agg)

**Description:**
1. Handles cases where the number of patches (n_patch) is less than n_agg by repeating patches to meet n_agg.
2. Pads patches with the last few patches if n_patch is not divisible by n_agg.
3. Splits patches into n_agg groups, concatenates them along the feature dimension, and projects to d_model using a linear layer with pre-normalization.


class EncoderLayer(torch.nn.Module)
------------------------------------

**The encoder layer**

A single encoder layer combining patch merging (optional) and BAModule for feature refinement.

.. code-block:: python

    __init__(self, n_agg=4, n_pool=10, factor=10, actv='entmax', hopfield=True, d_model=512, n_heads=8, d_ff=1024, dropout=0.2)

**Parameters:**

* **n_agg** *(int, optional, Default is 4)* - Number of patches to aggregate (1 means no merging).
* **n_pool** *(int, optional, Default is 10)* - Number of pooling vectors for BAModule.
* **factor** *(int, optional, Default is 10)* - Factor dimension for pooling vectors in BAModule.
* **actv** *(str, optional, Default is 'entmax')* - Activation function type for BAModule.
* **hopfield** *(bool, optional, Default is True)* - Whether to use Hopfield attention in BAModule.
* **d_model** *(int, optional, Default is 512)* - Number of features in each patch.
* **n_heads** *(int, optional, Default is 8)* - Number of attention heads in BAModule.
* **d_ff** *(int, optional, Default is 1024)* - Hidden dimension of feedforward networks in BAModule.
* **dropout** *(float, optional, Default is 0.2)* - Dropout probability for BAModule.


.. code-block:: python

    forward(self, x)

Processes input through optional patch merging and BAModule.

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor of shape [batch_size, emb_dim, n_patch, d_model]

**Returns:**

* **torch.Tensor** - Encoded tensor after patch merging (if enabled) and BAModule processing, shape [batch_size, emb_dim, processed_n_patch, d_model]


class Encoder(torch.nn.Module)
------------------------------

**Full encoder stack with multiple layers**

A stack of encoder layers that progressively processes patches, with optional patch merging in deeper layers.

.. code-block:: python

    __init__(self, e_layers=3, n_agg=4, d_model=512, n_heads=8, d_ff=1024, dropout=0.2, n_pool=10, factor=10, actv='entmax', hopfield=True)

**Parameters:**

* **e_layers** *(int, optional, Default is 3)* - Number of encoder layers.
* **n_agg** *(int, optional, Default is 4)* - Number of patches to aggregate in layers after the first.
* **d_model** *(int, optional, Default is 512)* - Number of features in each patch.
* **n_heads** *(int, optional, Default is 8)* - Number of attention heads in BAModule.
* **d_ff** *(int, optional, Default is 1024)* - Hidden dimension of feedforward networks in BAModule.
* **dropout** *(float, optional, Default is 0.2)* - Dropout probability for BAModule.
* **n_pool** *(int, optional, Default is 10)* - Number of pooling vectors for BAModule in the first layer.
* **factor** *(int, optional, Default is 10)* - Factor dimension for pooling vectors in BAModule.
* **actv** *(str, optional, Default is 'entmax')* - Activation function type for BAModule.
* **hopfield** *(bool, optional, Default is True)* - Whether to use Hopfield attention in BAModule.


.. code-block:: python

    forward(self, x)

Encodes input through a stack of encoder layers, capturing intermediate outputs.

**Parameters:**

* **x** *(torch.Tensor)* - Initial input tensor of shape [batch_size, emb_dim, initial_n_patch, d_model]

**Returns:**

* **list of torch.Tensor** - List of encoded tensors at each stage (including initial input). Each tensor has shape [batch_size, emb_dim, stage_n_patch, d_model], with stage_n_patch decreasing with deeper layers due to patch merging.

**Description:**
1. The first encoder layer does not perform patch merging (n_agg=1).
2. Subsequent layers apply patch merging with n_agg, reducing the number of patches progressively.
3. Collects and returns outputs from all stages (initial input + outputs after each encoder layer) for use in decoding.




**Core Utility Functions**
--------------------------

.. code-block:: python

    _make_ix_like(X, dim)

Generates an index tensor matching the specified dimension of the input tensor.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor used to determine the shape and device of the output.
* **dim** *(int)* - Dimension to match.

**Returns:**

* **torch.Tensor** - Index tensor with the same shape as X, where values along the specified dimension are 1, 2, ..., X.size(dim).


.. code-block:: python

    _roll_last(X, dim)

Moves the specified dimension to the last position of the tensor.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor.
* **dim** *(int)* - Dimension to move.

**Returns:**

* **torch.Tensor** - Tensor with the specified dimension moved to the last position.


.. code-block:: python

    _sparsemax_threshold_and_support(X, dim=-1, k=None)

Computes the optimal threshold and support size (number of non-zero elements) for Sparsemax.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor.
* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.
* **k** *(int or None, optional, Default is None)* - Number of largest elements to partially sort. If None, full sorting is performed.

**Returns:**

* **tau** *(torch.Tensor)* - Threshold tensor, with the same shape as X except for the specified dimension.
* **support_size** *(torch.LongTensor)* - Number of non-zero elements for each vector, with the same shape as tau.


.. code-block:: python

    _entmax_threshold_and_support(X, dim=-1, k=None)

Computes the optimal threshold and support size (number of non-zero elements) for 1.5-Entmax.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor.
* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.
* **k** *(int or None, optional, Default is None)* - Number of largest elements to partially sort. If None, full sorting is performed.

**Returns:**

* **tau_star** *(torch.Tensor)* - Threshold tensor, with the same shape as X except for the specified dimension.
* **support_size** *(torch.LongTensor)* - Number of non-zero elements for each vector, with the same shape as tau_star.


**Sparse Activation Function Classes**
---------------------------------------

class SparsemaxFunction(torch.autograd.Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autograd function implementing the Sparsemax transformation, a sparse normalization.

.. code-block:: python

    forward(cls, ctx, X, dim=-1, k=None)

Forward pass: Computes the Sparsemax transformation.

**Parameters:**

* **ctx** - Context object to store information for backward pass.
* **X** *(torch.Tensor)* - Input tensor.
* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.
* **k** *(int or None, optional, Default is None)* - Number of largest elements to partially sort.

**Returns:**

* **torch.Tensor** - Result of Sparsemax transformation, non-negative with sum 1 along the specified dimension.


.. code-block:: python

    backward(cls, ctx, grad_output)

Backward pass: Computes gradients for the input tensor.

**Parameters:**

* **ctx** - Context object containing stored information from forward pass.
* **grad_output** *(torch.Tensor)* - Gradient of the loss with respect to the output.

**Returns:**

* **torch.Tensor** - Gradient of the loss with respect to the input tensor X.


class Entmax15Function(torch.autograd.Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autograd function implementing the 1.5-Entmax transformation, a sparse normalization based on Tsallis entropy.

.. code-block:: python

    forward(cls, ctx, X, dim=0, k=None)

Forward pass: Computes the 1.5-Entmax transformation.

**Parameters:**

* **ctx** - Context object to store information for backward pass.
* **X** *(torch.Tensor)* - Input tensor.
* **dim** *(int, optional, Default is 0)* - Dimension along which to compute.
* **k** *(int or None, optional, Default is None)* - Number of largest elements to partially sort.

**Returns:**

* **torch.Tensor** - Result of 1.5-Entmax transformation, non-negative with sum 1 along the specified dimension.


.. code-block:: python

    backward(cls, ctx, dY)

Backward pass: Computes gradients for the input tensor.

**Parameters:**

* **ctx** - Context object containing stored information from forward pass.
* **dY** *(torch.Tensor)* - Gradient of the loss with respect to the output.

**Returns:**

* **torch.Tensor** - Gradient of the loss with respect to the input tensor X.


class Sparsemax(torch.nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module wrapping the Sparsemax transformation.

.. code-block:: python

    __init__(self, dim=-1, k=None)

**Parameters:**

* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.
* **k** *(int or None, optional, Default is None)* - Number of largest elements to partially sort.


.. code-block:: python

    forward(self, X)

Applies the Sparsemax transformation.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor.

**Returns:**

* **torch.Tensor** - Result of Sparsemax transformation.


class Entmax15(torch.nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module wrapping the 1.5-Entmax transformation.

.. code-block:: python

    __init__(self, dim=-1, k=None)

**Parameters:**

* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.
* **k** *(int or None, optional, Default is None)* - Number of largest elements to partially sort.


.. code-block:: python

    forward(self, X)

Applies the 1.5-Entmax transformation.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor.

**Returns:**

* **torch.Tensor** - Result of 1.5-Entmax transformation.


class AlphaChooser(torch.nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module for choosing alpha parameters in EntmaxAlpha, constraining alpha to (1, 2].

.. code-block:: python

    __init__(self, head_count)

**Parameters:**

* **head_count** *(int)* - Number of attention heads, determining the number of alpha parameters.


.. code-block:: python

    forward(self)

Computes and returns alpha parameters.

**Returns:**

* **torch.Tensor** - Alpha parameter tensor with shape [head_count], values in (1, 2].


class EntmaxAlpha(torch.nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module implementing Entmax with learnable alpha parameters.

.. code-block:: python

    __init__(self, head_count=1, dim=-1)

**Parameters:**

* **head_count** *(int, optional, Default is 1)* - Number of attention heads.
* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.


.. code-block:: python

    forward(self, att_scores)

Applies Entmax transformation to attention scores.

**Parameters:**

* **att_scores** *(torch.Tensor)* - Attention scores tensor with shape [batch_size, head_count, query_len, key_len].

**Returns:**

* **torch.Tensor** - Normalized attention weights with the same shape as att_scores.


class EntmaxBisectFunction(torch.autograd.Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autograd function implementing alpha-Entmax via bisection (root finding), supporting arbitrary alpha > 1.

.. code-block:: python

    forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True)

Forward pass: Computes the alpha-Entmax transformation.

**Parameters:**

* **ctx** - Context object to store information for backward pass.
* **X** *(torch.Tensor)* - Input tensor.
* **alpha** *(float or torch.Tensor, optional, Default is 1.5)* - Alpha parameter, must be > 1.
* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.
* **n_iter** *(int, optional, Default is 50)* - Number of bisection iterations.
* **ensure_sum_one** *(bool, optional, Default is True)* - Whether to ensure the result sums to 1 along the specified dimension.

**Returns:**

* **torch.Tensor** - Result of alpha-Entmax transformation.


.. code-block:: python

    backward(cls, ctx, dY)

Backward pass: Computes gradients for the input tensor and alpha parameter.

**Parameters:**

* **ctx** - Context object containing stored information from forward pass.
* **dY** *(torch.Tensor)* - Gradient of the loss with respect to the output.

**Returns:**

* **torch.Tensor** - Gradient of the loss with respect to the input tensor X.
* **torch.Tensor or None** - Gradient of the loss with respect to alpha (if required).


**Utility Functions**
---------------------

.. code-block:: python

    sparsemax(X, dim=-1, k=None)

Function interface for Sparsemax transformation.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor.
* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.
* **k** *(int or None, optional, Default is None)* - Number of largest elements to partially sort.

**Returns:**

* **torch.Tensor** - Result of Sparsemax transformation.


.. code-block:: python

    entmax15(X, dim=-1, k=None)

Function interface for 1.5-Entmax transformation.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor.
* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.
* **k** *(int or None, optional, Default is None)* - Number of largest elements to partially sort.

**Returns:**

* **torch.Tensor** - Result of 1.5-Entmax transformation.


.. code-block:: python

    entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True)

Function interface for alpha-Entmax transformation via bisection.

**Parameters:**

* **X** *(torch.Tensor)* - Input tensor.
* **alpha** *(float or torch.Tensor, optional, Default is 1.5)* - Alpha parameter, must be > 1.
* **dim** *(int, optional, Default is -1)* - Dimension along which to compute.
* **n_iter** *(int, optional, Default is 50)* - Number of bisection iterations.
* **ensure_sum_one** *(bool, optional, Default is True)* - Whether to ensure the result sums to 1 along the specified dimension.

**Returns:**

* **torch.Tensor** - Result of alpha-Entmax transformation.


.. code-block:: python

    ifnone(a, b)

Returns b if a is None, otherwise returns a.

**Parameters:**

* **a** - First value.
* **b** - Value to return if a is None.

**Returns:**

* a if a is not None else b.


**MLP and Full Model**
----------------------

class MLP(torch.nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-layer perceptron (MLP) layer for final prediction.

.. code-block:: python

    __init__(self, input_dim, output_dim, actv=None, bn=True, bn_final=False, dropout=0.2, hidden=(4, 2, 1), skip_connect=False, softmax=False)

**Parameters:**

* **input_dim** *(int)* - Input data dimension.
* **output_dim** *(int)* - Output data dimension (e.g., 1 for regression).
* **actv** *(torch.nn.Module or None, optional, Default is None)* - Activation function, default is ReLU.
* **bn** *(bool, optional, Default is True)* - Whether to use batch normalization in each layer.
* **bn_final** *(bool, optional, Default is False)* - Whether to use batch normalization in the final layer.
* **dropout** *(float, optional, Default is 0.2)* - Dropout ratio.
* **hidden** *(tuple, optional, Default is (4, 2, 1))* - Hidden layer configuration, each element is a multiple of the input dimension.
* **skip_connect** *(bool, optional, Default is False)* - Whether to add skip connections.
* **softmax** *(bool, optional, Default is False)* - If True and output_dim > 1, applies Softmax to the output.


.. code-block:: python

    forward(self, x)

MLP forward pass.

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor with shape [batch_size, input_dim].

**Returns:**

* **torch.Tensor** - Output tensor with shape [batch_size, output_dim].


class BAModel(torch.nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BAModel integrates patch embedding, encoder, and decoder to form a complete model.

.. code-block:: python

    __init__(self, feat_dim, emb_dim=32, out_dim=24, patch_dim=8, factor=8, n_agg=4, actv='entmax', hopfield=True, d_model=512, d_ff=1024, n_heads=8, e_layer=3, d_layer=4, dropout=0.2)

**Parameters:**

* **feat_dim** *(int)* - Input feature dimension.
* **emb_dim** *(int, optional, Default is 32)* - Embedding dimension.
* **out_dim** *(int, optional, Default is 24)* - Output dimension.
* **patch_dim** *(int, optional, Default is 8)* - Number of features per patch.
* **factor** *(int, optional, Default is 8)* - Factor dimension for pooling vectors.
* **n_agg** *(int, optional, Default is 4)* - Number of patches to aggregate.
* **actv** *(str, optional, Default is 'entmax')* - Type of activation function.
* **hopfield** *(bool, optional, Default is True)* - Whether to use Hopfield attention.
* **d_model** *(int, optional, Default is 512)* - Feature dimension for attention mechanisms.
* **d_ff** *(int, optional, Default is 1024)* - Hidden layer dimension for feedforward networks.
* **n_heads** *(int, optional, Default is 8)* - Number of attention heads.
* **e_layer** *(int, optional, Default is 3)* - Number of encoder layers.
* **d_layer** *(int, optional, Default is 4)* - Number of decoder layers.
* **dropout** *(float, optional, Default is 0.2)* - Dropout ratio.


.. code-block:: python

    forward(self, x)

Model forward pass.

**Parameters:**

* **x** *(torch.Tensor)* - Input tensor with shape [batch_size, feat_dim, emb_dim].

**Returns:**

* **torch.Tensor** - Model output tensor, shape depends on decoder configuration, ultimately [batch_size, out_dim, emb_dim] or similar.

**Referencses:**

Xu, C., Huang, Y.-C., Hu, J. Y.-C., Li, W., Gilani, A., Goan, H.-S., & Liu, H. (2024). BiSHop: Bi-Directional Cellular Learning for Tabular Data with Generalized Sparse Modern Hopfield Model. In Proceedings of the 41st International Conference on Machine Learning (ICML). `<https://arxiv.org/abs/2404.03830>`_