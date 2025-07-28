**TData**
=========================

A PyTorch Dataset implementation designed for handling tabular data with numerical and categorical features.



.. code-block:: python

    class TData(Dataset):
        def __init__(self, is_regression, X, Y, y_info, part):
            ...

**Parameters**

- **is_regression** *(bool)*: 
  Flag indicating if the task is regression. Used to determine label processing and metadata.
  
- **X** *(Tuple[Optional[Tensor], Optional[Tensor]])*: 
  Tuple containing numerical and categorical features. 
  - `X[0]`: Numerical features (shape: `[n_samples, n_num_features]`).
  - `X[1]`: Categorical features (shape: `[n_samples, n_cat_features]`).
  
- **Y** *(Dict[str, Tensor])*: 
  Dictionary with labels for train/val/test splits. Keys must include `part`.
  
- **y_info** *(Dict[str, Any])*: 
  Metadata about the labels. For regression, typically contains `mean` and `std` for de-normalization.
  
- **part** *(str)*: 
  Data split to use. Must be one of `['train', 'val', 'test']`.


**Core Methods**
----------------

**Feature Dimension Retrieval**

.. code-block:: python

    def get_dim_in(self) -> int:

**Returns**:

- **int**: Number of numerical features (`self.X_num.shape[1]`), or 0 if `self.X_num` is `None`.


**Categorical Feature Cardinality**

.. code-block:: python

    def get_categories(self) -> Optional[List[int]]:

**Returns**:

- **Optional[List[int]]**: 
  List where each element is the number of unique categories for a categorical feature. 
  Returns `None` if `self.X_cat` is `None`.


**Dataset Size**

.. code-block:: python

    def __len__(self) -> int:

**Returns**:

- **int**: Number of samples in the dataset (`len(self.Y)`).


**Sample Retrieval**

.. code-block:: python

    def __getitem__(self, i) -> Tuple[Union[Tensor, Tuple[Tensor, Tensor]], Tensor]:

**Parameters**:

- **i** *(int)*: Sample index.

**Returns**:

- **Tuple[Union[Tensor, Tuple[Tensor, Tensor]], Tensor]**:
  - **data**: Feature tensor(s):

    - If both numerical and categorical features exist: `(self.X_num[i], self.X_cat[i])`.
    - If only categorical features exist: `self.X_cat[i]`.
    - If only numerical features exist: `self.X_num[i]`.
    
  - **label**: Corresponding target label (`self.Y[i]`).


**Usage Notes**
---------------
1. **Data Structure**:
   - Expects numerical and categorical features to be provided as tensors (or `None` if not applicable).
   - Labels should be provided as a dictionary with keys 'train', 'val', 'test'.

2. **Feature Handling**:
   - Automatically handles datasets with numerical-only, categorical-only, or mixed features.
   - Categorical feature counts are computed using CPU conversion for set operations.

3. **Metadata**:
   - Stores label metadata (`y_info`) which can be used for tasks like de-normalizing regression predictions.

4. **Compatibility**:
   - Works with PyTorch DataLoader for efficient batching and parallel data loading.
   - The `get_categories` method is useful for initializing embeddings for categorical features.