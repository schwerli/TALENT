**DNNR (Distance-Weighted Nearest Regression)**
==========================================================

K-nearest neighbors (KNN) is one of the earliest and most established algorithms in machine learning. For regression tasks, KNN averages the targets within a neighborhood which poses a number of challenges: the neighborhood definition is crucial for the predictive performance as neighbors might be selected based on uninformative features, and averaging does not account for how the function changes locally. We propose a novel method called Differential Nearest Neighbors Regression (DNNR) that addresses both issues simultaneously: during training, DNNR estimates local gradients to scale the features; during inference, it performs an n-th order Taylor approximation using estimated gradients. In a large-scale evaluation on over 250 datasets, we find that DNNR performs comparably to state-of-the-art gradient boosting methods and MLPs while maintaining the simplicity and transparency of KNN. This allows us to derive theoretical error bounds and inspect failures. In times that call for transparency of ML models, DNNR provides a good balance between performance and interpretability.


**Data Classes for Predictions**
---------------------------------

@dataclasses.dataclass
class NeighborPrediction
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stores detailed prediction information from a single neighbor.

**Fields:**

* **neighbor_x** *(np.ndarray)* - Feature vector of the neighbor point.
* **neighbor_y** *(np.ndarray)* - Target value of the neighbor point.
* **neighbors_xs** *(np.ndarray)* - Feature vectors of the neighbor's own neighbors.
* **neighbors_ys** *(np.ndarray)* - Target values of the neighbor's own neighbors.
* **query** *(np.ndarray)* - The query point being predicted.
* **local_prediction** *(np.ndarray)* - Prediction for the query point based on this neighbor.
* **derivative** *(np.ndarray)* - Derivatives (e.g., gradients, Hessians) used in the prediction.
* **prediction_fn** *(Callable[[np.ndarray], np.ndarray])* - Function to generate predictions for new points using this neighbor's parameters.
* **intercept** *(Optional[np.ndarray], Default is None)* - Intercept term for the local prediction model (if applicable).


@dataclasses.dataclass
class DNNRPrediction
~~~~~~~~~~~~~~~~~~~~

Aggregates prediction results for a single query point.

**Fields:**
* **query** *(np.ndarray)* - The query point being predicted.
* **y_pred** *(np.ndarray)* - Final aggregated prediction for the query point.
* **neighbor_predictions** *(list[NeighborPrediction])* - Collection of predictions from individual neighbors.
* **y_true** *(Optional[np.ndarray], Default is None)* - Ground truth target value (if provided during analysis).


**Core DNNR Model**
-------------------

@dataclasses.dataclass
class DNNR(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary DNNR model class, implementing distance-weighted nearest neighbor regression with Taylor series approximations.

.. code-block:: python

    __init__(self, n_neighbors=3, n_derivative_neighbors=-1, order="1", fit_intercept=False, solver="linear_regression", index="annoy", index_kwargs=dataclasses.field(default_factory=dict), scaling="learned", scaling_kwargs=dataclasses.field(default_factory=dict), precompute_derivatives=False, clip=False)

**Parameters:**

* **n_neighbors** *(int, optional, Default is 3)* - Number of nearest neighbors to use for predicting each query point.
* **n_derivative_neighbors** *(int, optional, Default is -1)* - Number of neighbors used to approximate derivatives (gradients, Hessians). Defaults to `3 * input_dimension` if set to -1.
* **order** *(str, optional, Default is "1")* - Order of the Taylor series approximation:
  - "1": First-order approximation (uses gradients only)
  - "2diag": First-order + diagonal elements of second-order derivatives
  - "2": First-order + full second-order matrix (gradient and Hessian)
  - "3diag": First-order + diagonal elements of second and third-order derivatives
* **fit_intercept** *(bool, optional, Default is False)* - Whether to estimate an intercept term in the local prediction models.
* **solver** *(Union[str, Solver], optional, Default is "linear_regression")* - Method for solving linear systems to estimate derivatives. Can be a string identifier (e.g., "linear_regression", "ridge") or a `Solver` subclass instance.
* **index** *(Union[str, BaseIndex], optional, Default is "annoy")* - Type of nearest neighbor index to use. Options include "annoy" (Approximate Nearest Neighbors) or "kd_tree" (exact KD-Tree), or a custom `BaseIndex` subclass.
* **index_kwargs** *(dict, optional)* - Keyword arguments passed to the nearest neighbor index constructor.
* **scaling** *(Union[None, str, InputScaling], optional, Default is "learned")* - Strategy for input feature scaling:
  - None/"no_scaling": No scaling applied.
  - "learned": Scaling factors learned to maximize prediction performance.
* **scaling_kwargs** *(dict, optional)* - Keyword arguments passed to the scaling mechanism.
* **precompute_derivatives** *(bool, optional, Default is False)* - Whether to precompute derivatives for all training points during `fit` (speeds up prediction but increases memory usage).
* **clip** *(bool, optional, Default is False)* - Whether to clip predicted values to the range of training target values (`[min(y_train), max(y_train)]`).


.. code-block:: python

    fit(self, X_train: np.ndarray, y_train: np.ndarray) -> DNNR

Trains the DNNR model on training data.

**Parameters:**

* **X_train** *(np.ndarray)* - Training features with shape `(n_samples, n_features)`.
* **y_train** *(np.ndarray)* - Training target values with shape `(n_samples,)`.

**Returns:**

* **DNNR** - The fitted model instance.


.. code-block:: python

    predict(self, X_test: np.ndarray) -> np.ndarray

Generates predictions for test data.

**Parameters:**

* **X_test** *(np.ndarray)* - Test features with shape `(n_test_samples, n_features)`.

**Returns:**

* **np.ndarray** - Predicted target values with shape `(n_test_samples,)`.


.. code-block:: python

    point_analysis(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None) -> list[DNNRPrediction]

Performs in-depth analysis of predictions for individual test points, including breakdowns by neighbor.

**Parameters:**

* **X_test** *(np.ndarray)* - Test features with shape `(n_test_samples, n_features)`.
* **y_test** *(Optional[np.ndarray], Default is None)* - True target values for test points, with shape `(n_test_samples,)`.

**Returns:**

* **list[DNNRPrediction]** - Detailed prediction results for each test point.


**Input Scaling Mechanisms**
----------------------------

class InputScaling(sklearn.base.BaseEstimator, metaclass=abc.ABCMeta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base class for input feature scaling strategies.

.. code-block:: python

    fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> np.ndarray

Fits scaling parameters to training (and optionally validation) data.

**Parameters:**

* **X_train** *(np.ndarray)* - Training features.
* **y_train** *(np.ndarray)* - Training target values.
* **X_test** *(Optional[np.ndarray])* - Validation features (if used for fitting).
* **y_test** *(Optional[np.ndarray])* - Validation target values (if used for fitting).

**Returns:**

* **np.ndarray** - Learned scaling vector.


.. code-block:: python

    transform(self, X: np.ndarray) -> np.ndarray

Applies the learned scaling to input data.

**Parameters:**

* **X** *(np.ndarray)* - Input features to scale.

**Returns:**

* **np.ndarray** - Scaled features.


class NoScaling(InputScaling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A scaling strategy that applies no scaling (identity transformation).

.. code-block:: python

    fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> np.ndarray

Fits the scaling (returns a vector of ones).

**Returns:**

* **np.ndarray** - Vector of ones with shape `(n_features,)`.


.. code-block:: python

    transform(self, X: np.ndarray) -> np.ndarray

Returns the input data unchanged.

**Parameters:**

* **X** *(np.ndarray)* - Input features.

**Returns:**

* **np.ndarray** - Unmodified input features.


class LearnedScaling(InputScaling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A scaling strategy that learns feature scaling factors to optimize prediction performance via a cosine similarity objective.

.. code-block:: python

    __init__(self, n_epochs=1, optimizer=SGD, optimizer_params=dataclasses.field(default_factory=dict), shuffle=True, epsilon=1e-6, random=dataclasses.field(default_factory=lambda: random_mod.Random(random_mod.randint(0, 2**32 - 1))), show_progress=False, fail_on_nan=False, index='annoy', index_kwargs=dataclasses.field(default_factory=dict))

**Parameters:**

* **n_epochs** *(int, optional, Default is 1)* - Number of epochs to train the scaling factors.
* **optimizer** *(Union[str, Type[_Optimizer]], optional, Default is SGD)* - Optimization algorithm for learning scaling factors. Can be "sgd", "rmsprop", or a custom `_Optimizer` subclass.
* **optimizer_params** *(dict, optional)* - Hyperparameters for the optimizer (e.g., learning rate).
* **shuffle** *(bool, optional, Default is True)* - Whether to shuffle training data during optimization.
* **epsilon** *(float, optional, Default is 1e-6)* - Small value to prevent division by zero.
* **random** *(random_mod.Random)* - Random number generator for reproducibility.
* **show_progress** *(bool, optional, Default is False)* - Whether to display a progress bar during training.
* **fail_on_nan** *(bool, optional, Default is False)* - Whether to raise an error if NaN values appear in gradients.
* **index** *(Union[str, Type[BaseIndex]], optional, Default is 'annoy')* - Nearest neighbor index type used during scaling training.
* **index_kwargs** *(dict, optional)* - Keyword arguments for the nearest neighbor index.


**Optimizers**
--------------

@dataclasses.dataclass
class SGD(_Optimizer)
~~~~~~~~~~~~~~~~~~~~~

Stochastic Gradient Descent optimizer for updating parameters.

.. code-block:: python

    __init__(self, parameters: List[np.ndarray], lr: float = 0.01)

**Parameters:**

* **parameters** *(List[np.ndarray])* - List of parameters to optimize.
* **lr** *(float, optional, Default is 0.01)* - Learning rate.


.. code-block:: python

    step(self, gradients: List[np.ndarray]) -> None

Updates parameters using computed gradients.

**Parameters:**

* **gradients** *(List[np.ndarray])* - Gradients of the loss with respect to each parameter.


@dataclasses.dataclass
class RMSPROP(_Optimizer)
~~~~~~~~~~~~~~~~~~~~~~~~

RMSPROP optimizer, which adapts learning rates using a moving average of squared gradients.

.. code-block:: python

    __init__(self, parameters: List[np.ndarray], lr: float = 1e-4, γ: float = 0.99, eps: float = 1e-08)

**Parameters:**

* **parameters** *(List[np.ndarray])* - List of parameters to optimize.
* **lr** *(float, optional, Default is 1e-4)* - Learning rate.
* **γ** *(float, optional, Default is 0.99)* - Decay rate for the moving average of squared gradients.
* **eps** *(float, optional, Default is 1e-08)* - Small value to prevent division by zero.


.. code-block:: python

    step(self, gradients: List[np.ndarray]) -> None

Updates parameters using computed gradients and adaptive learning rates.

**Parameters:**

* **gradients** *(List[np.ndarray])* - Gradients of the loss with respect to each parameter.


**Nearest Neighbor Indices**
----------------------------

class BaseIndex(sklearn.base.BaseEstimator, metaclass=abc.ABCMeta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base class for nearest neighbor indexing structures.

.. code-block:: python

    fit(self, x: np.ndarray) -> None

Builds the index from input data.

**Parameters:**

* **x** *(np.ndarray)* - Data to index, with shape `(n_samples, n_features)`.


.. code-block:: python

    query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]

Retrieves the k nearest neighbors of a query point.

**Parameters:**

* **v** *(np.ndarray)* - Query point with shape `(n_features,)`.
* **k** *(int)* - Number of neighbors to retrieve.

**Returns:**

* **tuple[np.ndarray, np.ndarray]** - Tuple containing:
  - Indices of the k nearest neighbors (shape `(k,)`)
  - Distances to the k nearest neighbors (shape `(k,)`)


@dataclasses.dataclass
class KDTreeIndex(BaseIndex)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KD-Tree based index for exact nearest neighbor search (uses `sklearn.neighbors.KDTree`).

.. code-block:: python

    __init__(self, metric: str = "euclidean", leaf_size: int = 40, kwargs: dict[str, Any] = dataclasses.field(default_factory=dict))

**Parameters:**

* **metric** *(str, optional, Default is "euclidean")* - Distance metric to use (e.g., "euclidean", "manhattan").
* **leaf_size** *(int, optional, Default is 40)* - Size of leaves in the KD-Tree (affects memory and speed).
* **kwargs** *(dict, optional)* - Additional keyword arguments passed to `sklearn.neighbors.KDTree`.


@dataclasses.dataclass
class AnnoyIndex(BaseIndex)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Annoy (Approximate Nearest Neighbors Oh Yeah) index for fast approximate nearest neighbor search.

.. code-block:: python

    __init__(self, metric: str = "euclidean", n_trees: int = 50, n_features: Optional[int] = None)

**Parameters:**

* **metric** *(str, optional, Default is "euclidean")* - Distance metric to use.
* **n_trees** *(int, optional, Default is 50)* - Number of trees in the Annoy index (trades off speed and accuracy).
* **n_features** *(Optional[int])* - Dimensionality of input features (inferred from data during `fit` if None).


**Linear Solvers**
------------------

class Solver(ABC)
~~~~~~~~~~~~~~~~~

Abstract base class for solving linear systems to estimate derivatives.

.. code-block:: python

    solve(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray

Solves the weighted linear system `a^T * diag(w) * a * x = a^T * diag(w) * b`.

**Parameters:**

* **a** *(np.ndarray)* - Design matrix with shape `(n_samples, n_features)`.
* **b** *(np.ndarray)* - Target vector with shape `(n_samples,)`.
* **w** *(np.ndarray)* - Weights for each sample with shape `(n_samples,)`.

**Returns:**

* **np.ndarray** - Solution vector `x` with shape `(n_features,)`.


class SKLinearRegression(Solver)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solver using `sklearn.linear_model.LinearRegression` (ordinary least squares).


class SKRidge(Solver)
~~~~~~~~~~~~~~~~~~~~~

Solver using `sklearn.linear_model.Ridge` (ridge regression with L2 regularization).


class SKLasso(Solver)
~~~~~~~~~~~~~~~~~~~~~

Solver using `sklearn.linear_model.Lasso` (L1 regularization).


class ScipyLsqr(Solver)
~~~~~~~~~~~~~~~~~~~~~~~

Solver using `scipy.sparse.linalg.lsqr` (iterative least squares for sparse systems).


class NPSolver(Solver)
~~~~~~~~~~~~~~~~~~~~~~

Solver using numpy's pseudoinverse (`np.linalg.pinv`) for solving linear systems.


.. code-block:: python

    create_solver(solver: str) -> Solver

Creates a `Solver` instance from a string identifier.

**Parameters:**

* **solver** *(str)* - Name of the solver ("linear_regression", "ridge", "lasso", "scipy_lsqr", or "numpy").

**Returns:**

* **Solver** - Instance of the requested solver.


**Helper Functions**
--------------------

.. code-block:: python

    get_index_class(index: type[BaseIndex] | str) -> type[BaseIndex]

Retrieves the nearest neighbor index class corresponding to a string or class.

**Parameters:**

* **index** *(Union[str, type[BaseIndex]])* - Index name ("annoy" or "kd_tree") or a `BaseIndex` subclass.

**Returns:**

* **type[BaseIndex]** - Nearest neighbor index class.

**Referencses:**

DNNR: Differential Nearest Neighbors Regression
Youssef Nader, Leon Sixt, Tim Landgraf
In Proceedings of the 39th International Conference on Machine Learning (ICML 2022),
PMLR 162:16296–16317, 2022.
PDF <https://proceedings.mlr.press/v162/nader22a/nader22a.pdf>_
Project page <https://proceedings.mlr.press/v162/nader22a.html>_