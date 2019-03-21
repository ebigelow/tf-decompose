# tf-decompose

CP and Tucker tensor decompositions implemented in TensorFlow.


### Usage

```python
import numpy as np
import tensorflow as tf
from scipy.io.matlab import loadmat
from ktensor import KruskalTensor

# Load sensory bread data (http://www.models.life.ku.dk/datasets)
mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10,11,8])

# Build ktensor and learn CP decomposition using ALS with specified optimizer
T = KruskalTensor(X.shape, rank=3, regularize=1e-6, init='nvecs', X_data=X)
X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=20000)

# Save reconstructed tensor to file
np.save('X_predict.npy', X_predict)
```


## Kruskal tensors

<!--

*Input*: $\X \in \mathbb{R}^{d_1, ..., d_N}$
*Output*: $U_1 \in \mathbb{R}^{r \times d_1}


general problem:
    min_\X* ||\X - \X*||  


with:
    \X* = [\![  \lambda ;  U_1, ..., U_N  ]\!]
      &\triangleq sum_{r=1}^R \lambda_r  \lambda_r  \  u_1_r  \odot u_2_r \odot ... \odot u_N_r
      &= U_1 \Lambda ( U_N  \cdot  U_{N-1}  \cdot  U_{N-2} \cdot ...  \cdot  U_2 ) ^T


ALS formulation of problem:
    min_U_i* || X  -  U_i \Lambda (U_1 \cdot ... \cdot U_{n-1} \cdot U_{n+1} \cdot ... \cdot U_N)^T ||_F



 -->





### Notes on ALS gradient computation

- For CP decomposition we use alternating least squares' (ALS) over component matrices, but do not compute the exact solution as in Kolda & Bader (2009) due to the computational demands of computing large matrix inversions.
- In our tests we find inferior results to the exact solution descent method (requires inverting potentially huge matrices) implemented in `scikit-tensor` with ~.80 vs. ~.90 fit with decomposed rank-3 tensors on the Sensory Bread dataset.
- `tf-decompose` parallelized on GPU was approximately 20 times faster than `scikit-tensor` for a rank-200 decomposition of a random tensor with 60 million parameters.



## Tucker tensors

Preliminary results: with sensory bread data, `TuckerTensor.hosvd` seems to perform quite poorly, while `TuckerTensor.hooi` and `DecomposedTensor.train_als` learn reconstructions with fit ~0.70.



## References

Bader, Brett W., and Tamara G. Kolda. "Efficient MATLAB computations with sparse and factored tensors." SIAM Journal on Scientific Computing 30.1 (2007): 205-231.

Kolda, Tamara G., and Brett W. Bader. "Tensor decompositions and applications." SIAM review 51.3 (2009): 455-500.

Nickel, Maximilian. [`scikit-tensor`](https://github.com/mnick/scikit-tensor)

<br>

***Also see***: `tensorD` ([code](https://github.com/Large-Scale-Tensor-Decomposition/tensorD), [paper](https://www.sciencedirect.com/science/article/pii/S0925231218310178)). I wrote `tf-decompose` before this was available; I haven't used it, but you should check it out as well if you're considering using `tf-decompose`.
