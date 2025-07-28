**ExcelFormer**
================================================
Data organized in tabular format is ubiquitous in real-world applications, and users often craft tables with biased feature definitions and flexibly set prediction targets of their interests. Thus, a rapid development of a robust, effective, dataset-versatile, user-friendly tabular prediction approach is highly desired. While Gradient Boosting Decision Trees (GBDTs) and existing deep neural networks (DNNs) have been extensively utilized by professional users, they present several challenges for casual users, particularly: (i) the dilemma of model selection due to their different dataset preferences, and (ii) the need for heavy hyperparameter searching, failing which their performances are deemed inadequate. In this paper, we delve into this question: Can we develop a deep learning model that serves as a "sure bet" solution for a wide range of tabular prediction tasks, while also being user-friendly for casual users? We delve into three key drawbacks of deep tabular models, encompassing: (P1) lack of rotational variance property, (P2) large data demand, and (P3) over-smooth solution. We propose ExcelFormer, addressing these challenges through a semi-permeable attention module that effectively constrains the influence of less informative features to break the DNNs' rotational invariance property (for P1), data augmentation approaches tailored for tabular data (for P2), and attentive feedforward network to boost the model fitting capability (for P3). These designs collectively make ExcelFormer a "sure bet" solution for diverse tabular datasets. Extensive and stratified experiments conducted on real-world datasets demonstrate that our model outperforms previous approaches across diverse tabular data prediction tasks, and this framework can be friendly to casual users, offering ease of use without the heavy hyperparameter tuning.

**Feature Shuffling Augmentation**
----------------------------------

.. code-block:: python

    def batch_feat_shuffle(Xs: torch.Tensor, beta=0.5) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:

Applies feature-wise shuffling between random pairs of samples in a batch.

**Parameters:**

* **Xs** *(torch.Tensor)* - Input tensor with shape `(batch_size, features)` or `(batch_size, features, dim)`.
* **beta** *(float, optional, Default is 0.5)* - Shape parameter for the Beta distribution controlling the shuffling rate.

**Returns:**

* **Xs_mixup** *(torch.Tensor)* - Augmented tensor with randomly shuffled features.
* **feat_masks** *(torch.Tensor)* - Binary masks indicating which features were shuffled (shape `(batch_size, features)`).
* **shuffled_sample_ids** *(np.ndarray)* - Indices used for shuffling samples.

**Description:**

Randomly shuffles features between pairs of samples in the batch based on a Beta distribution. Each feature in each sample has a probability of being replaced by the corresponding feature from another randomly selected sample. This creates new synthetic samples that combine features from different original samples.


**Dimension-wise Shuffling Augmentation**
-----------------------------------------

.. code-block:: python

    def batch_dim_shuffle(Xs: torch.Tensor, beta=0.5) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:

Applies dimension-wise shuffling between random pairs of samples in a batch.

**Parameters:**

* **Xs** *(torch.Tensor)* - Input tensor with shape `(batch_size, features, dim)`.
* **beta** *(float, optional, Default is 0.5)* - Shape parameter for the Beta distribution controlling the shuffling rate.

**Returns:**

* **Xs_mixup** *(torch.Tensor)* - Augmented tensor with randomly shuffled dimensions.
* **shuffle_rates** *(torch.Tensor)* - Shuffling rates drawn from the Beta distribution (shape `(batch_size,)`).
* **shuffled_sample_ids** *(np.ndarray)* - Indices used for shuffling samples.

**Description:**

Randomly shuffles entire dimensions between pairs of samples in the batch. For each sample and each dimension, a random decision is made whether to replace the entire dimension with the corresponding dimension from another sample. This creates synthetic samples that combine dimensions from different original samples.


**Traditional Mixup Augmentation**
----------------------------------

.. code-block:: python

    def mixup_data(Xs: torch.Tensor, beta=0.5) -> Tuple[torch.Tensor, float, np.ndarray]:

Applies traditional mixup augmentation by linearly interpolating between pairs of samples.

**Parameters:**

* **Xs** *(torch.Tensor)* - Input tensor with shape `(batch_size, features)`.
* **beta** *(float, optional, Default is 0.5)* - Shape parameter for the Beta distribution controlling the interpolation coefficient.

**Returns:**

* **mixed_X** *(torch.Tensor)* - Augmented tensor created by mixing pairs of samples.
* **lam** *(float)* - Interpolation coefficient drawn from the Beta distribution.
* **shuffle_sample_ids** *(np.ndarray)* - Indices used for shuffling samples.

**Description:**

Creates new synthetic samples by linearly interpolating between pairs of samples using a coefficient drawn from a Beta distribution. For each sample, another sample is randomly selected, and the new sample is computed as:mixed_sample = λ * sample1 + (1 - λ) * sample2
where λ is drawn from Beta(β, β). This method encourages the model to learn linear combinations of features, improving generalization.

**Neural Network Utilities and Components**
===========================================

A collection of PyTorch modules, activation functions, optimization utilities, and helper functions for building and training neural networks.


**Normalization Layers**
------------------------

class Lambda(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~

A simple wrapper module to apply a custom function as a PyTorch module.

.. code-block:: python

    __init__(self, f: ty.Callable) -> None

**Parameters:**
* **f** *(Callable)* - A function to apply in the forward pass.


.. code-block:: python

    forward(self, x)

Applies the wrapped function to the input.

**Parameters:**
* **x** - Input tensor.

**Returns:**
* Output of the function applied to `x`.


class RMSNorm(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~

Root Mean Square Layer Normalization, a variant of layer normalization that normalizes inputs using the root mean square.

.. code-block:: python

    __init__(self, d, p=-1.0, eps=1e-5, bias=False)

**Parameters:**
* **d** *(int)* - Model dimension (input feature size).
* **p** *(float, optional, Default is -1.0)* - Fraction of features to use for partial RMSNorm (range [0, 1]; disabled if <0).
* **eps** *(float, optional, Default is 1e-5)* - Epsilon for numerical stability.
* **bias** *(bool, optional, Default is False)* - Whether to include a learnable bias term.


.. code-block:: python

    forward(self, x)

Applies RMS normalization to the input.

**Parameters:**
* **x** *(torch.Tensor)* - Input tensor with shape `(..., d)`.

**Returns:**
* **torch.Tensor** - Normalized tensor with the same shape as input.


class ScaleNorm(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Scale Normalization, a lightweight normalization that scales inputs by a learnable parameter divided by their norm.

.. code-block:: python

    __init__(self, d: int, eps: float = 1e-5, clamp: bool = False) -> None

**Parameters:**
* **d** *(int)* - Model dimension (used to initialize the scale parameter as `sqrt(d)`).
* **eps** *(float, optional, Default is 1e-5)* - Epsilon added to norms for stability.
* **clamp** *(bool, optional, Default is False)* - Whether to clamp norms to a minimum of `eps` (instead of adding `eps`).


.. code-block:: python

    forward(self, x)

Applies scale normalization to the input.

**Parameters:**
* **x** *(torch.Tensor)* - Input tensor with shape `(..., d)`.

**Returns:**
* **torch.Tensor** - Normalized tensor with the same shape as input.


**Activation Functions**
------------------------

.. code-block:: python

    reglu(x: Tensor) -> Tensor

ReLUGLU activation: splits input into two halves, applies ReLU to the second half, and returns their product.

**Parameters:**
* **x** *(torch.Tensor)* - Input tensor with even last dimension.

**Returns:**
* **torch.Tensor** - Output tensor with shape `(..., d/2)` where `d` is the input's last dimension.


.. code-block:: python

    geglu(x: Tensor) -> Tensor

GELUGLU activation: splits input into two halves, applies GELU to the second half, and returns their product.

**Parameters:**
* **x** *(torch.Tensor)* - Input tensor with even last dimension.

**Returns:**
* **torch.Tensor** - Output tensor with shape `(..., d/2)` where `d` is the input's last dimension.


.. code-block:: python

    tanglu(x: Tensor) -> Tensor

TanhGLU activation: splits input into two halves, applies Tanh to the second half, and returns their product.

**Parameters:**
* **x** *(torch.Tensor)* - Input tensor with even last dimension.

**Returns:**
* **torch.Tensor** - Output tensor with shape `(..., d/2)` where `d` is the input's last dimension.


class ReGLU(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~

Module wrapper for `reglu` activation.

.. code-block:: python

    forward(self, x: Tensor) -> Tensor

Applies `reglu` activation.

**Parameters:**
* **x** *(torch.Tensor)* - Input tensor.

**Returns:**
* **torch.Tensor** - Output of `reglu(x)`.


class GEGLU(nn.Module)
~~~~~~~~~~~~~~~~~~~~~~

Module wrapper for `geglu` activation.

.. code-block:: python

    forward(self, x: Tensor) -> Tensor

Applies `geglu` activation.

**Parameters:**
* **x** *(torch.Tensor)* - Input tensor.

**Returns:**
* **torch.Tensor** - Output of `geglu(x)`.


**Optimization Utilities**
--------------------------

.. code-block:: python

    make_optimizer(optimizer: str, parameter_groups, lr: float, weight_decay: float) -> optim.Optimizer

Creates an optimizer instance from a string identifier.

**Parameters:**
* **optimizer** *(str)* - Name of the optimizer (`adabelief`, `adam`, `adamw`, `radam`, `sgd`).
* **parameter_groups** - Parameters to optimize (typically from `model.parameters()`).
* **lr** *(float)* - Learning rate.
* **weight_decay** *(float)* - Weight decay (L2 penalty).

**Returns:**
* **optim.Optimizer** - Initialized optimizer.


.. code-block:: python

    make_lr_schedule(optimizer: optim.Optimizer, lr: float, epoch_size: int, lr_schedule: ty.Optional[ty.Dict[str, ty.Any]]) -> ty.Tuple[ty.Optional[optim.lr_scheduler._LRScheduler], ty.Dict[str, ty.Any], ty.Optional[int]]

Creates a learning rate scheduler.

**Parameters:**
* **optimizer** *(optim.Optimizer)* - Optimizer to schedule.
* **lr** *(float)* - Base learning rate.
* **epoch_size** *(int)* - Number of steps per epoch.
* **lr_schedule** *(Optional[Dict])* - Scheduler configuration (defaults to `{'type': 'constant'}`).

**Returns:**
* **Optional[optim.lr_scheduler._LRScheduler]** - Learning rate scheduler.
* **Dict** - Scheduler configuration.
* **Optional[int]** - Number of warmup steps (if applicable).


**Activation Function Helpers**
-------------------------------

.. code-block:: python

    get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]

Retrieves an activation function by name.

**Parameters:**
* **name** *(str)* - Name of the activation (`reglu`, `geglu`, `sigmoid`, `tanglu`, or any function in `torch.nn.functional`).

**Returns:**
* **Callable** - Activation function.


.. code-block:: python

    get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]

Retrieves the non-GLU counterpart of an activation (e.g., ReLU for ReGLU).

**Parameters:**
* **name** *(str)* - Name of the GLU activation (`reglu`, `geglu`, or any function in `torch.nn.functional`).

**Returns:**
* **Callable** - Non-GLU activation function.


**Training Utilities**
----------------------

.. code-block:: python

    load_swa_state_dict(model: nn.Module, swa_model: optim.swa_utils.AveragedModel)

Loads a Stochastic Weight Averaging (SWA) state dict into a model.

**Parameters:**

* **model** *(nn.Module)* - Model to load weights into.
* **swa_model** *(optim.swa_utils.AveragedModel)* - SWA model with averaged weights.


.. code-block:: python

    get_epoch_parameters(train_size: int, batch_size: ty.Union[int, str]) -> ty.Tuple[int, int]

Determines batch size and steps per epoch based on training data size.

**Parameters:**

* **train_size** *(int)* - Number of training samples.
* **batch_size** *(int or str)* - Batch size (or preset name: `v1`, `v2`, `v3`).

**Returns:**

* **int** - Batch size.
* **int** - Steps per epoch.


**Learning Rate Schedulers**
----------------------------

.. code-block:: python

    get_linear_warmup_lr(lr: float, n_warmup_steps: int, step: int) -> float

Computes learning rate for linear warmup.

**Parameters:**

* **lr** *(float)* - Base learning rate.
* **n_warmup_steps** *(int)* - Number of warmup steps.
* **step** *(int)* - Current step (1-based).

**Returns:**

* **float** - Warmup learning rate.


.. code-block:: python

    get_manual_lr(schedule: ty.List[float], epoch: int) -> float

Retrieves a manually specified learning rate for an epoch.

**Parameters:**

* **schedule** *(List[float])* - List of learning rates per epoch.
* **epoch** *(int)* - Current epoch (1-based).

**Returns:**

* **float** - Learning rate for the epoch.


.. code-block:: python

    get_transformer_lr(scale: float, d: int, n_warmup_steps: int, step: int) -> float

Computes learning rate using the Transformer schedule (Vaswani et al.).

**Parameters:**

* **scale** *(float)* - Scale factor.
* **d** *(int)* - Model dimension.
* **n_warmup_steps** *(int)* - Number of warmup steps.
* **step** *(int)* - Current step.

**Returns:**
* **float** - Transformer learning rate.


**Training Loop Helpers**
-------------------------

.. code-block:: python

    learn(model, optimizer, loss_fn, step, batch, star) -> ty.Tuple[Tensor, ty.Any]

Performs a single training step.

**Parameters:**

* **model** *(nn.Module)* - Model to train.
* **optimizer** *(optim.Optimizer)* - Optimizer.
* **loss_fn** - Loss function.
* **step** - Function to compute model output from a batch.
* **batch** - Input batch.
* **star** *(bool)* - Whether the loss function takes multiple arguments (from `step` output).

**Returns:**
* **Tensor** - Loss value.
* **Any** - Model output.


**Model Utilities**
-------------------

.. code-block:: python

    tensor(x) -> torch.Tensor

Asserts and casts input to a PyTorch tensor.

**Parameters:**
* **x** *(torch.Tensor)* - Input to cast.

**Returns:**
* **torch.Tensor** - Input as a tensor.


.. code-block:: python

    get_n_parameters(m: nn.Module)

Counts the number of trainable parameters in a model.

**Parameters:**
* **m** *(nn.Module)* - Model to inspect.

**Returns:**
* **int** - Number of trainable parameters.


.. code-block:: python

    get_mlp_n_parameters(units: ty.List[int])

Counts parameters in an MLP with given layer sizes.

**Parameters:**
* **units** *(List[int])* - List of MLP layer sizes (input to output).

**Returns:**
* **int** - Total number of parameters.


**Optimizer Helpers**
---------------------

.. code-block:: python

    get_lr(optimizer: optim.Optimizer) -> float

Gets the current learning rate from an optimizer.

**Parameters:**
* **optimizer** *(optim.Optimizer)* - Optimizer.

**Returns:**
* **float** - Current learning rate.


.. code-block:: python

    set_lr(optimizer: optim.Optimizer, lr: float) -> None

Sets the learning rate for all parameter groups in an optimizer.

**Parameters:**
* **optimizer** *(optim.Optimizer)* - Optimizer.
* **lr** *(float)* - New learning rate.


**Device Utilities**
--------------------

.. code-block:: python

    get_device() -> torch.device

Gets the default device (CUDA if available, else CPU).

**Returns:**
* **torch.device** - Default device.


**Gradient Utilities**
----------------------

.. code-block:: python

    get_gradient_norm_ratios(m: nn.Module)

Computes the ratio of gradient norms to parameter norms for all parameters.

**Parameters:**
* **m** *(nn.Module)* - Model to inspect.

**Returns:**
* **Dict** - Mapping from parameter names to gradient/parameter norm ratios.


**Error Handling**
------------------

.. code-block:: python

    is_oom_exception(err: RuntimeError) -> bool

Checks if a runtime error is due to out-of-memory (OOM).

**Parameters:**
* **err** *(RuntimeError)* - Error to check.

**Returns:**
* **bool** - True if the error is OOM-related.


**Custom Optimizers**
---------------------

class RAdam(optim.Optimizer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rectified Adam optimizer, a variant of Adam with improved convergence properties.

.. code-block:: python

    __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True)

**Parameters:**
* **params** - Parameters to optimize.
* **lr** *(float, optional, Default is 1e-3)* - Learning rate.
* **betas** *(Tuple[float, float], optional, Default is (0.9, 0.999))* - Momentum parameters.
* **eps** *(float, optional, Default is 1e-8)* - Epsilon for stability.
* **weight_decay** *(float, optional, Default is 0)* - Weight decay.
* **degenerated_to_sgd** *(bool, optional, Default is True)* - Whether to fall back to SGD for unstable cases.


class AdaBelief(optim.Optimizer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AdaBelief optimizer, which adapts stepsizes based on "belief" in observed gradients.

.. code-block:: python

    __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=True, degenerated_to_sgd=True, print_change_log=True)
、
**Parameters:**
* **params** - Parameters to optimize.
* **lr** *(float, optional, Default is 1e-3)* - Learning rate.
* **betas** *(Tuple[float, float], optional, Default is (0.9, 0.999))* - Momentum parameters.
* **eps** *(float, optional, Default is 1e-16)* - Epsilon for stability.
* **weight_decay** *(float, optional, Default is 0)* - Weight decay.
* **amsgrad** *(bool, optional, Default is False)* - Whether to use AMSGrad variant.
* **weight_decouple** *(bool, optional, Default is True)* - Whether to use decoupled weight decay.
* **fixed_decay** *(bool, optional, Default is False)* - Whether weight decay is fixed (not scaled by lr).
* **rectify** *(bool, optional, Default is True)* - Whether to use rectified updates (like RAdam).
* **degenerated_to_sgd** *(bool, optional, Default is True)* - Whether to fall back to SGD for unstable cases.
* **print_change_log** *(bool, optional, Default is True)* - Whether to print configuration changes.


.. code-block:: python

    reset(self)

Resets the optimizer state (exponential moving averages and step count).


.. code-block:: python

    step(self, closure=None)

Performs a single optimization step for AdaBelief.

**Parameters:**
* **closure** *(callable, optional)* - A closure that reevaluates the model and returns the loss.

**Returns:**
* **float or None** - Loss value if closure is provided, else None.

**Description:**
Implements the AdaBelief optimization algorithm, which adapts step sizes based on the "belief" in observed gradients (measured by the variance of gradient residuals). Supports features like decoupled weight decay, rectified updates (similar to RAdam), and AMSGrad for stable convergence.


**Additional Notes**
--------------------
- **RAdam**: Addresses the convergence issues of Adam in early training stages by rectifying the adaptive learning rate using the variance of gradient moments.
- **AdaBelief**: Extends Adam by incorporating gradient uncertainty (variance of residuals) into step size calculation, improving generalization in tasks like computer vision and NLP.
- Both optimizers include fallbacks to SGD for unstable scenarios, ensuring robustness across different training regimes.

**Referencses:**

ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data
Jintai Chen, Jiahuan Yan, Qiyuan Chen, Danny Ziyi Chen, Jian Wu, Jimeng Sun
arXiv preprint arXiv:2301.02819, 2024.
 `<https://arxiv.org/abs/2301.02819>`_