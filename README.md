# ThinkfulCapstone

![sine](http://i.giphy.com/l3q2s3MQ4bPb5RogU.gif)

Bayesian optimization is an optimization method that can be used in the following situations:

* Your objective function is an unknown function (a computer simulation or physical experiment)
* The objective function is expensive to evaluate (long computer experiment/ costly laboratory analysis)
* No gradient information is easily available.
* Global optima are wanted rather than local.

pyGPGO is a simple and modular Python (>3.5) package for bayesian optimization.  I am using this package
as the codebase for my ensemble modeling changes.

My forked code adds the ability to ensemble Gaussian Processes or even Random Forests and Boosted Trees
as the surrogate function.

* Should reduce the possibility of model regret for expensive test runs.
* Reduce model variance and improve prediction should lead to improved optimization.

### Installation

Retrieve my fork from the repo,

```bash
pip install git+https://github.com/dataronio/pyGPGO
```

Check our documentation in http://pygpgo.readthedocs.io/.

All dependencies are automatically taken care for in the requirements file.

### An Example of Using an Ensemble Gaussian Process!

The user only has to define a function to maximize and a dictionary specifying input space.

```python
import numpy as np
from pyGPGO.covfunc import matern32
from pyGPGO.covfunc import matern52
from pyGPGO.covfunc import gammaExponential
from pyGPGO.covfunc import expSine
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.Ensemble import Ensemble
from pyGPGO.covfunc import squaredExponential
from pyGPGO.covfunc import rationalQuadratic


def f(x, y):
    # Franke's function (https://www.mathworks.com/help/curvefit/franke.html)
    one = 0.75 * np.exp(-(9 * x - 2) ** 2 / 4 - (9 * y - 2) ** 2 / 4)
    two = 0.75 * np.exp(-(9 * x + 1) ** 2/ 49 - (9 * y + 1) / 10)
    three = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4 - (9 * y -3) ** 2 / 4)
    four = 0.25 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return one + two + three - four

np.random.seed(42)
mat32 = matern32()
mat52 = matern52()
gamex = gammaExponential()
sqex = squaredExponential()
rquad = rationalQuadratic()
gpmat32 = GaussianProcess(mat32,optimize=False, usegrads=False)
gpmat52 = GaussianProcess(mat52,optimize=False, usegrads=False)
gpsqex = GaussianProcess(sqex,optimize=False)
gprquad = GaussianProcess(rquad,optimize=False)
gpgamex = GaussianProcess(gamex,optimize=False)
acqEI = Acquisition(mode='ExpectedImprovement')
param = {'x': ('cont', [0, 1]),
         'y': ('cont', [0, 1])}
ens = Ensemble([gprquad,gpsqex,gpmat32,gpgamex,gpmat52])
gpgo = GPGO(ens, acqEI, franke, param)
gpgo.run(max_iter=100)

```

Check the `tutorials` and `examples` folders for more ideas on how to use the software.
