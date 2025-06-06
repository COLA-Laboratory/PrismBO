.. _register-new-algorithm:

Registering a New Algorithmic Object in PrismBO
===============================================

To register a new algorithm object in PrismBO, follow the steps outlined below:

1. **Import the Model Registry**

   First, you need to import the `model_registry` from the `prismbo.agent.registry` module:

   .. code-block:: python

      from prismbo.agent.registry import model_registry

2. **Define the Algorithm Object Name**

   Next, use the registry to define the name of your algorithm object. For example:

   .. code-block:: python

      @model_registry.register("MHGP")
      class MHGP(Model):
          pass

   In this example, the algorithm object is named "MHGP".

3. **Choose the Appropriate Base Class**

   Depending on the type of algorithm object you are creating, you must inherit from a specific base class. PrismBO provides several algorithm modules, each corresponding to a different base class:

   - **Surrogate Model**: Inherit from the `Model` class.
   - **Initialization Design**: Inherit from the `Sampler` class.
   - **Acquisition Function**: Inherit from the `AcquisitionBase` class.
   - **Pretrain Module**: Inherit from the `PretrainBase` class.
   - **Normalizer Module**: Inherit from the `NormalizerBase` class.

   For instance, in the example provided, we are creating a surrogate model, so the `MHGP` class inherits from the `Model` base class.

4. **Implement the Required Abstract Methods**

   Once the class is defined, you need to implement several abstract methods that are required by the `Model` base class. These methods include:

   .. code-block:: python

      def meta_fit(
          self,
          source_X : List[np.ndarray],
          source_Y : List[np.ndarray],
          optimize: Union[bool, Sequence[bool]] = True,
      ):
          pass

      def fit(
          self,
          X: np.ndarray,
          Y: np.ndarray,
          optimize: bool = False,
      ):
          pass

      def predict(
          self, X: np.ndarray, return_full: bool = False, with_noise: bool = False
      ) -> Tuple[np.ndarray, np.ndarray]:
          pass

   - **meta_fit**: This method is used to fit meta-data. If your transfer optimization algorithm requires meta-data, this is where you should leverage it.
   - **fit**: This method is used to fit the data for the current task.

By following these steps, you can successfully register a new algorithm object in PrismBO and implement the necessary functionality to integrate it into the framework.