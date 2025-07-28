**TabM**
=========================================

A model based on MLP and variations of BatchEnsemble.


**Memory Management**
---------------------

.. code-block:: python

    def is_oom_exception(err: RuntimeError) -> bool

Checks if a RuntimeError is caused by CUDA out-of-memory conditions.

**Parameters:**
* **err** *(RuntimeError)* - The exception to check.

**Returns:**
* **bool** - True if the error is a CUDA OOM error, False otherwise.


**Tensor Initialization**
-------------------------

.. code-block:: python

    def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor

Initializes a tensor with values sampled from a uniform distribution scaled by the reciprocal square root of `d`.

**Parameters:**
* **x** *(torch.Tensor)* - Tensor to initialize.
* **d** *(int)* - Scaling factor (typically input dimension).

**Returns:**
* **torch.Tensor** - Initialized tensor.


.. code-block:: python

    def init_random_signs_(x: Tensor) -> Tensor

Initializes a tensor with random +1/-1 values.

**Parameters:**

* **x** *(torch.Tensor)* - Tensor to initialize.

**Returns:**

* **torch.Tensor** - Tensor with random signs.


**Basic Modules**
-----------------

**Identity**
~~~~~~~~~~~~

A module that returns its input unchanged.

**Parameters:**

* **args** - Ignored.
* **kwargs** - Ignored.


**Mean**
~~~~~~~~

Computes the mean along a specified dimension.

**Parameters:**

* **dim** *(int)* - Dimension to compute the mean over.


**Ensemble Modules**
--------------------

**ScaleEnsemble**
~~~~~~~~~~~~~~~~~

Scales input tensors element-wise with learnable weights.

**Parameters:**

* **k** *(int)* - Number of ensemble members.
* **d** *(int)* - Feature dimension.
* **init** *(str)* - Initialization method ('ones', 'normal', 'random-signs').


**ElementwiseAffineEnsemble**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies element-wise affine transformation with learnable weights and biases.

**Parameters:**

* **k** *(int)* - Number of ensemble members.
* **d** *(int)* - Feature dimension.
* **weight** *(bool)* - Whether to include weight parameters.
* **bias** *(bool)* - Whether to include bias parameters.
* **weight_init** *(str)* - Weight initialization method.


**LinearEfficientEnsemble**
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient implementation of BatchEnsemble linear layer with configurable scaling and bias.

**Parameters:**

* **in_features** *(int)* - Input dimension.
* **out_features** *(int)* - Output dimension.
* **bias** *(bool)* - Whether to include bias.
* **k** *(int)* - Number of ensemble members.
* **ensemble_scaling_in** *(bool)* - Whether to scale inputs.
* **ensemble_scaling_out** *(bool)* - Whether to scale outputs.
* **ensemble_bias** *(bool)* - Whether to use ensemble-specific biases.
* **scaling_init** *(str)* - Scaling initialization method.


**OneHotEncoding0d**
~~~~~~~~~~~~~~~~~~~~

Performs one-hot encoding for categorical features, handling out-of-vocabulary values.

**Parameters:**

* **cardinalities** *(list[int])* - List of category cardinalities for each feature.

**Input Shape:**
`(*, n_cat_features)` where `n_cat_features` is the number of categorical features.

**Output Shape:**
`(*, sum(cardinalities))` where each feature is one-hot encoded.

**Note:**
- Handles out-of-vocabulary values by encoding them as all zeros.
- Assumes categorical values are encoded such that unknown categories are assigned the maximum possible index for that feature.

.. code-block:: python

    @torch.inference_mode()
    def _init_scaling_by_sections(
        weight: Tensor,
        distribution: Literal['normal', 'random-signs'],
        init_sections: list[int],
    ) -> None:

Initializes a weight tensor in sections, where all elements within a section share the same initial value. This is typically used for scaling parameters in efficient ensemble models, ensuring consistent initialization within feature sections.

**Parameters:**

* **weight** *(Tensor)* - 2D tensor to initialize, with shape `(num_ensemble_members, total_features)`.
* **distribution** *(Literal['normal', 'random-signs'])* - Distribution to use for initialization:
  - 'normal': Initializes using a normal distribution via `nn.init.normal_`.
  - 'random-signs': Initializes with random ±1 values via `init_random_signs_`.
* **init_sections** *(list[int])* - List of integers specifying the size of each section. The sum of sections must equal the total number of features (`weight.shape[1]`).

**Returns:**

* **None** - Modifies the input tensor `weight` in-place.

**Behavior:**

1. **Section Boundaries**: Computes cumulative sums of `init_sections` to determine the start and end indices of each section.
2. **Per-Section Initialization**: For each section:
   - Creates a 1D tensor of shape `(num_ensemble_members, 1)` initialized from the specified distribution.
   - Assigns this tensor to the corresponding section in `weight`, ensuring all elements within the section share the same value.

**Notes:**

- Requires `weight` to be a 2D tensor.
- Ensures `sum(init_sections) == weight.shape[1]` (asserted).
- Runs in inference mode to disable gradient tracking during initialization.
- Uses `init_random_signs_` for 'random-signs' distribution (assumed to be imported).

##References##

   Yury Gorishniy, Akim Kotelnikov, and Artem Babenko.
   *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*.
   `arXiv:2410.24210 <https://arxiv.org/abs/2410.24210>`_, 2025.