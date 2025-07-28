Methods
========

Deep learning method implementations that wrap model architectures with training logic.

This section contains method classes that provide a unified interface for training, validation, and prediction across all deep learning models in TALENT. Each method class inherits from the base `Method` class and implements model-specific logic while maintaining consistent APIs.

.. automodule:: TALENT.model.methods
   :members:
   :undoc-members:
   :show-inheritance:

Base Method Class
------------------

.. class:: Method
   :noindex:

   Abstract base class that provides a unified interface for all deep learning methods in TALENT.
   
   **Key Features:**
   
   * Consistent training/validation/prediction workflow
   * Automatic data preprocessing and formatting
   * Model construction and optimization setup
   * Early stopping and checkpoint management
   * Comprehensive evaluation metrics
   
   **Core Methods:**
   
   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the method with configuration and task type.
      
      **Parameters:**
      
      * **args** (*argparse.Namespace*) -- Configuration arguments
      * **is_regression** (*bool*) -- Whether the task is regression
   
   .. method:: construct_model(model_config=None)
      :noindex:
      
      Abstract method to construct the specific model architecture.
      Must be implemented by each method subclass.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model-specific configuration

   .. method:: data_format(is_train=True, N=None, C=None, y=None)
      :noindex:
      
      Format and preprocess data for training or inference.
      
      **Parameters:**
      
      * **is_train** (*bool*) -- Whether formatting for training or inference
      * **N** (*dict, optional*) -- Numerical features dictionary
      * **C** (*dict, optional*) -- Categorical features dictionary  
      * **y** (*dict, optional*) -- Target labels dictionary
      
      **Processing Pipeline:**
      
      **Training Mode (`is_train=True`):**
      
      1. **NaN Processing:** Handle missing values using specified policies
      2. **Label Processing:** Encode/normalize target variables
      3. **Numerical Encoding:** Apply feature encoding (PLE, Unary, etc.)
      4. **Categorical Encoding:** Apply categorical encoding (ordinal, one-hot, etc.)
      5. **Normalization:** Apply feature normalization
      6. **DataLoader Creation:** Create training and validation dataloaders
      
      **Inference Mode (`is_train=False`):**
      
      1. **Apply Fitted Transformers:** Use previously fitted encoders/normalizers
      2. **Create Test DataLoader:** Prepare data for inference
      3. **Feature Formatting:** Ensure compatibility with trained model

   .. method:: train_epoch(epoch)
      :noindex:
      
      Train the model for one epoch.
      
      **Parameters:**
      
      * **epoch** (*int*) -- Current epoch number
      
      **Training Process:**
      
      1. **Set Training Mode:** `model.train()`
      2. **Batch Processing:** Iterate through training batches
      3. **Forward Pass:** Compute model predictions
      4. **Loss Computation:** Calculate training loss
      5. **Backward Pass:** Compute gradients
      6. **Parameter Update:** Apply optimizer step
      7. **Progress Logging:** Track and display training progress

   .. method:: validate(epoch)
      :noindex:
      
      Validate the model on validation set.
      
      **Parameters:**
      
      * **epoch** (*int*) -- Current epoch number
      
      **Validation Process:**
      
      1. **Set Evaluation Mode:** `model.eval()`
      2. **Inference:** Process validation batches without gradients
      3. **Metric Computation:** Calculate validation metrics
      4. **Best Model Tracking:** Save model if performance improved
      5. **Early Stopping:** Stop training if no improvement for 20 epochs
      6. **Logging:** Record validation results

   .. method:: metric(predictions, labels, y_info)
      :noindex:
      
      Compute comprehensive evaluation metrics.
      
      **Parameters:**
      
      * **predictions** (*np.ndarray*) -- Model predictions
      * **labels** (*np.ndarray*) -- Ground truth labels
      * **y_info** (*dict*) -- Label processing information
      
      **Returns:**
      
      * **tuple** -- (metrics, metric_names)
      
      **Task-Specific Metrics:**
      
      **Regression Tasks:**
      
      * **MAE:** Mean Absolute Error
      * **R²:** Coefficient of determination  
      * **RMSE:** Root Mean Squared Error
      
      **Binary Classification:**
      
      * **Accuracy:** Overall classification accuracy
      * **Balanced Recall:** Balanced accuracy score
      * **Macro Precision:** Macro-averaged precision
      * **F1 Score:** Binary F1 score
      * **Log Loss:** Cross-entropy loss
      * **AUC:** Area under ROC curve
      
      **Multi-class Classification:**
      
      * **Accuracy:** Overall classification accuracy
      * **Balanced Recall:** Balanced accuracy score
      * **Macro Precision:** Macro-averaged precision
      * **Macro F1:** Macro-averaged F1 score
      * **Log Loss:** Cross-entropy loss
      * **Macro AUC:** One-vs-Rest AUC

Utility Functions
-----------------

.. function:: check_softmax(logits)
   :noindex:

   Check if logits are probabilities and convert if necessary.
   
   **Parameters:**
   
   * **logits** (*np.ndarray*) -- Array of shape (N, C) with logits or probabilities
   
   **Returns:**
   
   * **np.ndarray** -- Properly normalized probabilities
   
   **Mathematical Process:**
   
   1. **Probability Check:** Verify if values are in [0,1] and sum to 1
   2. **Softmax Application:** If not probabilities, apply softmax:
      
      .. math::
         
         p_i = \frac{\exp(x_i - \max(x))}{\sum_{j} \exp(x_j - \max(x))}
   
   3. **Numerical Stability:** Subtract max before exp to prevent overflow

Basic Neural Network Methods
----------------------------

.. class:: MLPMethod
   :noindex:

   Method class for Multi-Layer Perceptron (MLP) models.
   
   **Features:**
   
   * Simple feedforward neural network
   * Configurable hidden layers and dropout
   * Suitable for most tabular data tasks
   * Fast training and inference
   
   **Requirements:**
   
   * `cat_policy` cannot be 'indices' (categorical features must be encoded)
   
   **Model Configuration:**
   
   * **d_layers** (*List[int]*) -- Hidden layer dimensions
   * **dropout** (*float*) -- Dropout probability

.. class:: ResNetMethod
   :noindex:

   Method class for Residual Networks adapted for tabular data.
   
   **Features:**
   
   * Skip connections to prevent gradient vanishing
   * Multiple activation functions (ReLU, GELU, ReGLU, GeGLU)
   * Batch normalization or layer normalization
   * Deep architecture support
   
   **Model Configuration:**
   
   * **d_hidden** (*int*) -- Hidden dimension
   * **n_layers** (*int*) -- Number of residual blocks
   * **dropout** (*float*) -- Dropout probability
   * **activation** (*str*) -- Activation function type

.. class:: SNNMethod
   :noindex:

   Method class for Self-Normalizing Networks (SNN).
   
   **Features:**
   
   * SELU activation for self-normalization
   * Suitable for deeper networks
   * Automatic normalization properties
   * Fast convergence

Transformer-Based Methods
-------------------------

.. class:: FTTMethod
   :noindex:

   Method class for Feature Tokenizer Transformer (FT-Transformer).
   
   **Features:**
   
   * Feature tokenization for tabular data
   * Multi-head self-attention mechanism
   * State-of-the-art performance on many datasets
   * Configurable transformer architecture
   
   **Requirements:**
   
   * `cat_policy` must be 'indices' (uses raw categorical indices)
   
   **Model Configuration:**
   
   * **n_blocks** (*int*) -- Number of transformer blocks
   * **attention** -- Attention mechanism configuration
   * **ffn_dropout** (*float*) -- Feed-forward network dropout
   * **attention_dropout** (*float*) -- Attention dropout

.. class:: SaintMethod
   :noindex:

   Method class for Self-Attention and Intersample Attention Transformer (SAINT).
   
   **Features:**
   
   * Row and column attention mechanisms
   * Enhanced feature interaction modeling
   * Token-based representation
   * Improved performance on complex datasets

.. class:: TabTransformerMethod
   :noindex:

   Method class for TabTransformer.
   
   **Features:**
   
   * Column-wise attention for categorical features
   * Contextual embeddings
   * Strong performance on datasets with categorical features
   * Interpretable attention weights

Advanced Tabular Methods
------------------------

.. class:: TabNetMethod
   :noindex:

   Method class for TabNet with sequential attention.
   
   **Features:**
   
   * Sequential feature selection
   * Interpretable decision making
   * Sparse feature selection
   * Custom training workflow with TabNet-specific optimizers
   
   **Requirements:**
   
   * `cat_policy` cannot be 'indices' (requires encoded categorical features)
   
   **Custom Data Processing:**
   
   TabNet uses its own specialized data processing pipeline that differs from the base class:
   
   1. **Direct Processing:** Bypasses standard data_format method
   2. **TabNet-specific Encoding:** Uses TabNet's internal categorical handling
   3. **Custom DataLoaders:** Creates TabNet-compatible data structures
   
   **Model Configuration:**
   
   * **n_steps** (*int*) -- Number of decision steps
   * **gamma** (*float*) -- Relaxation parameter
   * **n_independent** (*int*) -- Number of independent GLU layers
   * **n_shared** (*int*) -- Number of shared GLU layers
   * **momentum** (*float*) -- Momentum for batch normalization

.. class:: TabRMethod
   :noindex:

   Method class for TabR (KNN-attention hybrid model).
   
   **Features:**
   
   * Combines KNN with attention mechanisms
   * Context-aware predictions
   * Strong performance on various datasets
   * Retrieval-based learning
   
   **Special Training Process:**
   
   TabR implements a unique training approach that combines retrieval and attention:
   
   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Custom fit method with retrieval-based training.
      
      **Special Features:**
      
      * **Context Management:** Maintains context_size=96 for retrieval
      * **Index Management:** Tracks training indices for efficient retrieval
      * **Candidate Selection:** Uses full training set as retrieval candidates
      
      **Training Process:**
      
      1. **Setup Retrieval Context:** Initialize training indices and context size
      2. **Model Construction:** Build TabR with attention and KNN components
      3. **Training Loop:** Standard epoch-based training with retrieval context
      4. **Candidate Updates:** Dynamically update retrieval candidates

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Prediction with retrieval-based inference.
      
      **Retrieval Process:**
      
      1. **Load Training Data:** Use training set as retrieval candidates
      2. **Neighbor Search:** Find relevant training examples
      3. **Attention Computation:** Apply attention over retrieved candidates
      4. **Prediction Generation:** Combine retrieval and learned representations

.. class:: ModernNCAMethod
   :noindex:

   Method class for Modern Nearest Class Analysis.
   
   **Features:**
   
   * Neighborhood Component Analysis-inspired
   * Embedding-based predictions
   * Effective for datasets with clear class structure
   * Interpretable neighbor relationships
   
   **Special Training Features:**
   
   * **Neighbor Sampling:** Efficient sampling of training neighbors
   * **Embedding Learning:** Learn optimal embedding space for distance computation
   * **Distance-based Loss:** Optimize for neighborhood classification

Specialized Methods
-------------------

.. class:: ProtoGateMethod
   :noindex:

   Method class for ProtoGate (prototype-based gating).
   
   **Features:**
   
   * Prototype-based learning
   * Gating mechanisms for feature selection
   * Suitable for high-dimensional low-sample-size data
   * Enhanced interpretability through prototypes
   
   **Prototype Learning:**
   
   * **Prototype Initialization:** Learn representative prototypes from data
   * **Gating Computation:** Use prototypes to gate feature importance
   * **Selection Mechanism:** Adaptive feature selection based on prototypes

.. class:: TromptMethod
   :noindex:

   Method class for Trompt (prompt-based tabular learning).
   
   **Features:**
   
   * Prompt-based neural architecture
   * Separation of intrinsic and sample-specific features
   * Multiple learning cycles for improved performance
   * Custom training with repeated targets
   
   **Requirements:**
   
   * `cat_policy` must be 'indices'
   
   **Custom Training Process:**
   
   .. method:: train_epoch(epoch)
      :noindex:
      
      Custom training with prompt-based learning.
      
      **Prompt Learning Process:**
      
      1. **Forward for Training:** Use `model.forward_for_training()` instead of standard forward
      2. **Multiple Cycles:** Model outputs multiple prediction cycles
      3. **Target Repetition:** Repeat targets for each cycle
      4. **Cycle Loss:** Compute loss across all cycles
      
      **Mathematical Formulation:**
      
      For n_cycles prediction cycles:
      
      .. math::
         
         \text{output} = \text{model.forward_for_training}(X_{num}, X_{cat}) \\
         \text{output} = \text{output.view}(-1, d_{out}) \\
         y_{repeated} = y.\text{repeat_interleave}(n_{cycles})

.. class:: ExcelFormerMethod
   :noindex:

   Method class for ExcelFormer with mixup training.
   
   **Features:**
   
   * Semi-permeable attention mechanisms
   * Multiple mixup strategies (feat_mix, hidden_mix, naive_mix)
   * Mutual information-based feature selection
   * Enhanced generalization through mixup
   
   **Custom Training Process:**
   
   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Enhanced fit method with mutual information preprocessing.
      
      **Preprocessing Steps:**
      
      1. **MI Score Computation:** Calculate mutual information scores between features and targets
      2. **Feature Ranking:** Sort features by MI scores for mixup weighting
      3. **Mixup Configuration:** Setup mixup parameters based on configuration
      
   .. method:: train_epoch(epoch)
      :noindex:
      
      Custom training with mixup strategies.
      
      **Mixup Training Process:**
      
      **No Mixup (`mix_type='none'`):**
      
      .. math::
         
         \text{loss} = \text{criterion}(\text{model}(X_{num}, X_{cat}, \text{mix_up}=False), y)
      
      **Feature Mixup (`mix_type='feat_mix'`):**
      
      .. math::
         
         \lambda = \sum (\text{MI_scores} \odot \text{feat_masks}) \\
         \text{loss} = \lambda \cdot \text{criterion}(\text{pred}, y) + (1-\lambda) \cdot \text{criterion}(\text{pred}, y[\text{shuffled_ids}])
      
      **Hidden Mixup (`mix_type='hidden_mix'`):**
      
      .. math::
         
         \text{loss} = \text{feat_masks} \cdot \text{criterion}(\text{pred}, y) + (1-\text{feat_masks}) \cdot \text{criterion}(\text{pred}, y[\text{shuffled_ids}])
   
   **Mixup Types:**
   
   * **none** -- No mixup augmentation
   * **feat_mix** -- Feature-level mixing weighted by mutual information scores
   * **hidden_mix** -- Hidden representation mixing
   * **naive_mix** -- Simple input-level mixing

Tree-Based Neural Methods
-------------------------

.. class:: NodeMethod
   :noindex:

   Method class for Neural Oblivious Decision Ensembles (NODE).
   
   **Features:**
   
   * Neural implementation of oblivious decision trees
   * Ensemble learning approach
   * Interpretable decision boundaries
   * Effective for structured data

.. class:: GrowNetMethod
   :noindex:

   Method class for GrowNet (gradient boosting with neural networks).
   
   **Features:**
   
   * Neural network weak learners
   * Gradient boosting framework
   * Dynamic model growth
   * Strong performance on various tasks

.. class:: GrandeMethod
   :noindex:

   Method class for GRANDE (gradient-boosted neural decision ensembles).
   
   **Features:**
   

   * Neural decision tree ensembles
   * Gradient descent optimization
   * Interpretable tree-like decisions
   * Enhanced performance through ensembling

High-Performance Methods
------------------------

.. class:: HyperFastMethod
   :noindex:

   Method class for HyperFast networks.
   
   **Features:**
   
   * Ultra-fast training and inference
   * Optimized architecture for speed
   * Meta-learning capabilities
   * Suitable for real-time applications

.. class:: RealMLPMethod
   :noindex:

   Method class for RealMLP (enhanced MLP).
   
   **Features:**
   
   * Improved MLP architecture
   * Better numerical stability
   * Enhanced performance on real-valued data
   * Efficient implementation

.. class:: SwitchTabMethod
   :noindex:

   Method class for SwitchTab (switch transformer for tabular data).
   
   **Features:**
   
   * Switch transformer architecture
   * Mixture of experts approach
   * Self-supervised learning components
   * Enhanced model capacity through expert routing

Foundation Model Methods
-------------------------

.. class:: TabPFNMethod
   :noindex:

   Method class for TabPFN (prior-free neural networks).
   
   **Features:**
   
   * Pre-trained foundation model for tabular data
   * Zero-shot or few-shot learning capabilities
   * No gradient-based training required
   * Fast inference on new datasets
   
   **Special Characteristics:**
   
   * **Pre-trained Weights:** Uses pre-trained model weights
   * **No Training:** Typically used without additional training
   * **Context Learning:** Learns from context examples
   * **Fast Deployment:** Immediate application to new datasets

.. class:: TabICLMethod
   :noindex:

   Method class for TabICL (in-context learning for tabular data).
   
   **Features:**
   
   * In-context learning capabilities
   * Foundation model approach
   * Meta-learning from diverse tabular datasets
   * Adaptive to new domains without fine-tuning

Method Usage Patterns
---------------------

**Basic Usage:**

.. code-block:: python
   
   from TALENT.model.utils import get_method
   
   # Get method class
   MethodClass = get_method('mlp')
   
   # Initialize method
   method = MethodClass(args, is_regression=True)
   
   # Train the method
   time_cost = method.fit(train_data, info)
   
   # Make predictions
   loss, metrics, metric_names, predictions = method.predict(test_data, info, 'best-val')

**Advanced Usage with Custom Training:**

.. code-block:: python
   
   # For methods with custom training (e.g., Trompt, ExcelFormer)
   method = TromptMethod(args, is_regression=False)
   
   # These methods override train_epoch for specialized training
   time_cost = method.fit(train_data, info)

**Retrieval-Based Methods:**

.. code-block:: python
   
   # TabR uses retrieval-based training
   tabr_method = TabRMethod(args, is_regression=True)
   
   # Automatically handles candidate selection and context management
   time_cost = tabr_method.fit(train_data, info)

**Method Selection Guidelines:**

* **Beginners:** Start with `MLPMethod` or `ResNetMethod`
* **Best Performance:** Try `FTTMethod`, `TabNetMethod`, or `ModernNCAMethod`
* **Interpretability:** Use `TabNetMethod`, `NodeMethod`, or `ProtoGateMethod`
* **Speed:** Choose `HyperFastMethod`, `SNNMethod`, or `MLPMethod`
* **Complex Features:** Consider `SaintMethod`, `TabTransformerMethod`, or `ExcelFormerMethod`
* **Foundation Models:** Use `TabPFNMethod` or `TabICLMethod` for quick deployment
* **Retrieval-Based:** Use `TabRMethod` or `ModernNCAMethod` for similarity-based learning

**Common Method Requirements:**

* **Categorical Policy:** Some methods require specific `cat_policy` settings
* **Data Preprocessing:** All methods inherit consistent preprocessing from base class
* **Model Configuration:** Each method accepts model-specific configuration parameters
* **Training Workflow:** All methods follow the same training/validation/prediction interface

**Error Handling and Debugging:**

* **Configuration Validation:** All methods validate configuration parameters
* **Data Compatibility:** Automatic checks for data format compatibility
* **Memory Management:** Efficient memory usage patterns across all methods
* **Progress Monitoring:** Built-in progress tracking and logging 