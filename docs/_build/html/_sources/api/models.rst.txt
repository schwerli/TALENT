====================================
Models
====================================

Deep learning models for tabular data, implementing various state-of-the-art architectures.

This section contains all the neural network architectures implemented in TALENT, ranging from simple MLPs to advanced transformer-based models specifically designed for tabular data. Each model implements specific forward pass computations and mathematical operations.

.. automodule:: TALENT.model.models
   :members:
   :undoc-members:
   :show-inheritance:

Basic Neural Networks
---------------------

Multi-Layer Perceptron (MLP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: MLP
   :noindex:

   Simple feedforward neural network with multiple fully connected layers.
   
   **Mathematical Formulation:**
   
   For input :math:`x \in \mathbb{R}^{d_{in}}`, the MLP computes:
   
   .. math::
      
      h_0 &= x \\
      h_i &= \text{ReLU}(\text{Linear}(h_{i-1})) = \text{ReLU}(W_i h_{i-1} + b_i) \\
      \text{output} &= W_{\text{head}} h_L + b_{\text{head}}
   
   where :math:`L` is the number of hidden layers.
   
   .. method:: forward(x, x_cat=None)
      :noindex:
      
      Forward pass through the MLP.
      
      **Parameters:**
      
      * **x** (*torch.Tensor*) -- Input numerical features of shape (batch_size, d_in)
      * **x_cat** (*torch.Tensor, optional*) -- Categorical features (not used in MLP)
      
      **Returns:**
      
      * **torch.Tensor** -- Output predictions of shape (batch_size, d_out) or (batch_size,) for regression
      
      **Mathematical Implementation:**
      
      .. code-block:: python
         
         for layer in self.layers:
             x = layer(x)  # Linear: x = W @ x + b
             x = F.relu(x)  # ReLU: x = max(0, x)
             if self.dropout:
                 x = F.dropout(x, self.dropout, self.training)
      
      The ReLU activation function:
      
      .. math::
         
         \text{ReLU}(x) = \max(0, x)
      
      For the final output:
      
      .. math::
         
         \text{logit} = W_{\text{head}} \cdot h_L + b_{\text{head}}
      
      If single output (regression), the tensor is squeezed:
      
      .. math::
         
         \text{output} = \text{logit.squeeze(-1)} \text{ if } d_{out} = 1

Residual Network (ResNet)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: ResNet
   :noindex:

   Deep residual network with skip connections for tabular data.
   
   **Mathematical Formulation:**
   
   ResNet uses residual blocks with skip connections:
   
   .. math::
      
      h_{i+1} = h_i + F(h_i, W_i)
   
   where :math:`F(h_i, W_i)` is the residual function.
   
   .. method:: forward(x, x_cat=None)
      :noindex:
      
      Forward pass through the ResNet.
      
      **Residual Block Mathematical Implementation:**
      
      For each residual block, the computation follows:
      
      .. math::
         
         \text{residual} &= \text{Norm}(h_i) \\
         \text{residual} &= \text{Linear}(\text{residual}) \\
         \text{residual} &= \text{Activation}(\text{residual}) \\
         \text{residual} &= \text{Dropout}(\text{residual}) \\
         h_{i+1} &= h_i + \text{residual}
      
      **Activation Functions:**
      
      * **ReLU:** :math:`\text{ReLU}(x) = \max(0, x)`
      * **GELU:** :math:`\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]`

Self-Normalizing Network (SNN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: SNN
   :noindex:

   Lightweight neural network with self-normalizing properties using SELU activation.
   
   .. method:: forward(x, x_cat=None)
      :noindex:
      
      Forward pass with SELU activation for self-normalization.
      
      **SELU Activation Mathematical Definition:**
      
      .. math::
         
         \text{SELU}(x) = \lambda \begin{cases}
         x & \text{if } x > 0 \\
         \alpha(e^x - 1) & \text{if } x \leq 0
         \end{cases}
      
      where :math:`\lambda \approx 1.0507` and :math:`\alpha \approx 1.6733`.
      
      **Self-Normalization Property:**
      
      SELU ensures that for normalized inputs:
      - Mean converges to 0
      - Variance converges to 1

Transformer-Based Models
------------------------

Feature Tokenizer Transformer (FT-Transformer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: Transformer
   :noindex:

   Advanced transformer architecture specifically designed for tabular data.
   
   **Mathematical Formulation:**
   
   **Feature Tokenization:**
   
   For numerical features: :math:`t_i^{\text{num}} = W_{\text{num}} x_i + b_{\text{num}}`
   
   For categorical features: :math:`t_i^{\text{cat}} = \text{Embedding}(x_i^{\text{cat}})`
   
   .. method:: forward(x_num, x_cat)
      :noindex:
      
      Forward pass through the transformer.
      
      **Parameters:**
      
      * **x_num** (*torch.Tensor, optional*) -- Numerical features
      * **x_cat** (*torch.Tensor, optional*) -- Categorical features
      
      **Transformer Layer Mathematical Implementation:**
      
      For each transformer layer:
      
      .. math::
         
         \text{attn_out} &= \text{MultiHeadAttention}(x, x, x) \\
         x &= \text{LayerNorm}(x + \text{attn_out}) \\
         \text{ffn_out} &= \text{FFN}(x) \\
         x &= \text{LayerNorm}(x + \text{ffn_out})

MultiheadAttention Module
~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: MultiheadAttention
   :noindex:

   Multi-head attention mechanism for transformer models.
   
   .. method:: forward(x_q, x_kv, key_compression, value_compression)
      :noindex:
      
      Compute multi-head attention.
      
      **Parameters:**
      
      * **x_q** (*torch.Tensor*) -- Query input
      * **x_kv** (*torch.Tensor*) -- Key and value input
      * **key_compression** (*nn.Linear, optional*) -- Key compression layer
      * **value_compression** (*nn.Linear, optional*) -- Value compression layer
      
      **Multi-Head Attention Mathematical Implementation:**
      
      1. **Linear Projections:**
         
         .. math::
            
            Q = x_q W^Q, \quad K = x_{kv} W^K, \quad V = x_{kv} W^V
      
      2. **Attention Score Computation:**
         
         .. math::
            
            \text{attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
      
      3. **Output Computation:**
         
         .. math::
            
            \text{output} = \text{attention} \cdot V
      
      **Multi-Head Formulation:**
      
      .. math::
         
         \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O

Advanced Tabular Models
-----------------------

TabNet
~~~~~~

.. class:: TabNetClassifier
   :noindex:

   Interpretable deep learning model with sequential attention mechanism.
   
   **Mathematical Formulation:**
   
   TabNet uses sequential feature selection through sparsemax attention:
   
   **Feature Selection at Step i:**
   
   .. math::
      
      M^{[i]} = \text{sparsemax}(\text{AttentionTransformer}(f^{[i-1]}))
   
   **Feature Processing:**
   
   .. math::
      
      f^{[i]} = \gamma \odot M^{[i]} \odot h + (1-\gamma) \odot f^{[i-1]}
   
   where :math:`\gamma` is the relaxation parameter.
   
   .. method:: predict_proba(X)
      :noindex:
      
      Make probability predictions for classification.
      
      **Parameters:**
      
      * **X** (*torch.Tensor or scipy.sparse matrix*) -- Input features
      
      **Returns:**
      
      * **np.ndarray** -- Class probabilities of shape (n_samples, n_classes)
      
      **Softmax Application:**
      
      .. math::
         
         P(y=k|x) = \frac{\exp(o_k)}{\sum_{j=1}^K \exp(o_j)}
      
      where :math:`o_k` is the raw output for class :math:`k`.

.. class:: TabNetRegressor
   :noindex:

   TabNet for regression tasks.
   
   .. method:: compute_loss(y_pred, y_true)
      :noindex:
      
      Compute mean squared error loss.
      
      **MSE Loss Mathematical Definition:**
      
      .. math::
         
         \mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2

GRANDE (Gradient-Boosted Neural Decision Ensembles)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: GRANDE
   :noindex:

   Tree-mimic neural network using gradient descent for decision tree simulation.
   
   **Mathematical Formulation:**
   
   GRANDE simulates decision trees using neural operations with entmax for sparse selection.
   
   .. method:: forward(inputs)
      :noindex:
      
      Forward pass through the GRANDE model.
      
      **Parameters:**
      
      * **inputs** (*torch.Tensor*) -- Input features
      
      **Tree Simulation Mathematical Implementation:**
      
      1. **Split Decision Computation:**
         
         .. math::
            
            \text{node_result} = \frac{\text{softsign}(s_1 - s_2) + 1}{2}
         
         where :math:`s_1` are split values and :math:`s_2` are feature values.
      
      2. **Path Probability Calculation:**
         
         .. math::
            
            p = \prod_{j} ((1-\text{path_id}_j) \cdot \text{node_result}_j + \text{path_id}_j \cdot (1-\text{node_result}_j))
      
      3. **Ensemble Output for Regression:**
         
         .. math::
            
            \text{output} = \sum_{e,l} w_e \cdot p_{e,l} \cdot v_{e,l}
         
         where :math:`w_e` are estimator weights, :math:`p_{e,l}` are leaf probabilities, and :math:`v_{e,l}` are leaf values.

Modern Nearest Class Analysis (ModernNCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: ModernNCA
   :noindex:

   Neighborhood Component Analysis-inspired model for tabular data.
   
   **Mathematical Formulation:**
   
   ModernNCA learns embeddings for distance-based classification.
   
   .. method:: forward(x, y, candidate_x, candidate_y, is_train)
      :noindex:
      
      Forward pass with neighborhood analysis.
      
      **Parameters:**
      
      * **x** (*torch.Tensor*) -- Query features
      * **y** (*torch.Tensor*) -- Query labels
      * **candidate_x** (*torch.Tensor*) -- Candidate features for nearest neighbor search
      * **candidate_y** (*torch.Tensor*) -- Candidate labels
      * **is_train** (*bool*) -- Training mode flag
      
      **Distance-Based Prediction Mathematical Implementation:**
      
      1. **Distance Computation:**
         
         .. math::
            
            d(x_i, x_j) = ||f(x_i) - f(x_j)||_2
         
         where :math:`f` is the learned embedding function.
      
      2. **Probability Assignment:**
         
         .. math::
            
            p_{ij} = \frac{\exp(-d(x_i, x_j))}{\sum_{k \neq i} \exp(-d(x_i, x_k))}
      
      3. **Final Prediction:**
         
         .. math::
            
            \hat{y}_i = \sum_j p_{ij} y_j

Specialized Architectures
-------------------------

ExcelFormer (Semi-Permeable Attention)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: ExcelFormer
   :noindex:

   Transformer with semi-permeable attention and mixup training capabilities.
   
   .. method:: forward(x_num, x_cat, mix_up, beta, mtype)
      :noindex:
      
      Forward pass with optional mixup augmentation.
      
      **Parameters:**
      
      * **x_num** (*torch.Tensor*) -- Numerical features
      * **x_cat** (*torch.Tensor, optional*) -- Categorical features
      * **mix_up** (*bool*) -- Whether to apply mixup
      * **beta** (*float*) -- Mixup parameter (default: 0.5)
      * **mtype** (*str*) -- Mixup type ('feat_mix', 'hidden_mix', 'naive_mix')
      
      **Mixup Mathematical Implementation:**
      
      **Feature Mixup:**
      
      .. math::
         
         \tilde{x} = \lambda x_i + (1-\lambda) x_j
      
      **Semi-Permeable Attention:**
      
      .. math::
         
         \text{Attention}_{\text{perm}}(Q, K, V) = \text{mask} \odot \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

ProtoGate (Prototype-Based Gating)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: GatingNet
   :noindex:

   Gating network for prototype-based feature selection.
   
   .. method:: hard_sigmoid(x)
      :noindex:
      
      Hard sigmoid activation for efficient gating.
      
      **Hard Sigmoid Mathematical Definition:**
      
      .. math::
         
         \text{hard_sigmoid}(x) = \max(0, \min(1, \frac{x + 1}{2}))
      
      This provides a piecewise linear approximation to the sigmoid function for computational efficiency.

Regularization Methods
----------------------

TANGOS Regularization
~~~~~~~~~~~~~~~~~~~~~

.. class:: Tangos
   :noindex:

   MLP with TANGOS regularization for neuron specialization.
   
   **Mathematical Formulation:**
   
   TANGOS applies spatial and spectral regularization to encourage neuron specialization:
   
   .. math::
      
      \mathcal{L}_{\text{TANGOS}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{spatial}} + \lambda_2 \mathcal{L}_{\text{spectral}}
   
   .. method:: cal_representation(x)
      :noindex:
      
      Calculate intermediate representations for regularization.
      
      **Parameters:**
      
      * **x** (*torch.Tensor*) -- Input features
      
      **Returns:**
      
      * **torch.Tensor** -- Hidden representations before final layer
      
      **Representation Extraction:**
      
      The method extracts intermediate representations by stopping before the final layer:
      
      .. code-block:: python
         
         for i, layer in enumerate(self.layers):
             x = layer(x)
             x = F.relu(x)
             if self.dropout and i != len(self.layers) - 1:
                 x = F.dropout(x, self.dropout, self.training)

Activation Functions Reference
------------------------------

**ReLU (Rectified Linear Unit):**

.. math::
   
   \text{ReLU}(x) = \max(0, x)

**GELU (Gaussian Error Linear Unit):**

.. math::
   
   \text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]

**SELU (Scaled Exponential Linear Unit):**

.. math::
   
   \text{SELU}(x) = \lambda \begin{cases}
   x & \text{if } x > 0 \\
   \alpha(e^x - 1) & \text{if } x \leq 0
   \end{cases}

**Softmax:**

.. math::
   
   \text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^K \exp(x_j)}

**Sparsemax (used in TabNet):**

.. math::
   
   \text{sparsemax}(z) = \arg\min_{p \in \Delta^{K-1}} ||p - z||_2^2

where :math:`\Delta^{K-1}` is the probability simplex.

Model Usage Examples
--------------------

**Basic MLP Usage:**

.. code-block:: python
   
   import torch
   from TALENT.model.models.mlp import MLP
   
   # Initialize MLP
   model = MLP(
       d_in=10,           # Input dimension
       d_out=3,           # Output dimension (3 classes)
       d_layers=[64, 32], # Hidden layer sizes
       dropout=0.1        # Dropout probability
   )
   
   # Forward pass
   x = torch.randn(32, 10)  # Batch of 32 samples, 10 features
   output = model(x)        # Shape: (32, 3)
   print(f"Output shape: {output.shape}")

**ResNet with Different Activations:**

.. code-block:: python
   
   from TALENT.model.models.resnet import ResNet
   
   # Initialize ResNet with GELU activation
   model = ResNet(
       d_in=15,
       d_out=1,              # Regression task
       d_hidden=128,
       n_layers=4,
       activation='gelu',
       normalization='layer_norm'
   )
   
   # Forward pass for regression
   x = torch.randn(64, 15)
   output = model(x)  # Shape: (64,) for regression

**FT-Transformer with Mixed Features:**

.. code-block:: python
   
   from TALENT.model.models.ftt import Transformer
   
   # Initialize FT-Transformer
   model = Transformer(
       d_numerical=8,          # 8 numerical features
       categories=[5, 10, 3],  # 3 categorical features with cardinalities
       d_token=64,
       n_layers=3,
       n_heads=8,
       d_ffn_factor=2.0
   )
   
   # Prepare mixed input
   x_num = torch.randn(32, 8)    # Numerical features
   x_cat = torch.randint(0, 5, (32, 3))  # Categorical features (adjust for cardinalities)
   
   output = model(x_num, x_cat)

**TabNet for Classification:**

.. code-block:: python
   
   from TALENT.model.models.tabnet import TabNetClassifier
   import numpy as np
   
   # Initialize TabNet
   model = TabNetClassifier(
       n_steps=3,
       gamma=1.3,
       n_independent=2,
       n_shared=2,
       momentum=0.02
   )
   
   # Prepare data
   X_train = np.random.random((1000, 20))
   y_train = np.random.randint(0, 3, 1000)
   
   # Fit the model
   model.fit(X_train, y_train)
   
   # Predict probabilities
   X_test = np.random.random((100, 20))
   probabilities = model.predict_proba(X_test)
   print(f"Predicted probabilities shape: {probabilities.shape}")

**GRANDE for Tree-like Decisions:**

.. code-block:: python
   
   from TALENT.model.models.grande import GRANDE
   
   # Initialize GRANDE
   model = GRANDE(
       batch_size=64,
       task_type='regression',
       depth=4,              # Tree depth
       n_estimators=10,      # Number of trees
       dropout=0.1
   )
   
   # Forward pass
   x = torch.randn(64, 12)
   output = model(x)

**ModernNCA with Candidate Selection:**

.. code-block:: python
   
   from TALENT.model.models.modernNCA import ModernNCA
   
   # Initialize ModernNCA
   model = ModernNCA(
       d_in=15,
       d_out=4,  # 4 classes
       k=32      # Number of nearest neighbors
   )
   
   # Training forward pass
   x = torch.randn(32, 15)
   y = torch.randint(0, 4, (32,))
   candidate_x = torch.randn(100, 15)  # Candidate pool
   candidate_y = torch.randint(0, 4, (100,))
   
   output = model(x, y, candidate_x, candidate_y, is_train=True)

**ExcelFormer with Mixup:**

.. code-block:: python
   
   from TALENT.model.models.excelformer import ExcelFormer
   
   # Initialize ExcelFormer
   model = ExcelFormer(
       d_numerical=10,
       d_token=64,
       n_blocks=3,
       attention_dropout=0.1,
       ffn_dropout=0.1
   )
   
   # Forward pass with feature mixup
   x_num = torch.randn(32, 10)
   output, feat_masks, shuffled_ids = model(
       x_num, 
       mix_up=True, 
       beta=0.5, 
       mtype='feat_mix'
   )

**Custom Loss with TANGOS Regularization:**

.. code-block:: python
   
   from TALENT.model.models.tangos import Tangos
   
   # Initialize TANGOS
   model = Tangos(
       d_in=20,
       d_out=5,
       d_layers=[128, 64],
       dropout=0.2,
       lambda1=0.1,  # Spatial regularization weight
       lambda2=0.1   # Spectral regularization weight
   )
   
   # Forward pass and representation extraction
   x = torch.randn(32, 20)
   output = model(x)
   representations = model.cal_representation(x)  # For regularization
   
   # Custom training loop would use both output and representations
   print(f"Output shape: {output.shape}")
   print(f"Representations shape: {representations.shape}")

Model Selection Guidelines
--------------------------

**Performance-Oriented Models:**

* **FT-Transformer:** Best overall performance, attention-based
* **TabNet:** Interpretable with good performance
* **ModernNCA:** Strong on many datasets, embedding-based

**Speed-Oriented Models:**

* **MLP:** Fastest training and inference
* **SNN:** Lightweight with self-normalization
* **ResNet:** Good balance of speed and performance

**Interpretability-Focused Models:**

* **TabNet:** Sequential attention provides interpretability
* **GRANDE:** Tree-like decision process
* **ProtoGate:** Prototype-based explanations

**Specialized Use Cases:**

* **TANGOS:** When regularization is important
* **ExcelFormer:** For complex feature interactions with mixup
* **ModernNCA:** When similarity-based predictions are desired 