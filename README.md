# Unraveling the latent geometry of network-driven processes

Complex systems are characterized by emergent patterns created by the non-trivial interplay between dynamical processes and the networks of interactions on which these processes unfold.Topological or dynamical descriptors alone are not enough to fully embrace this interplay in all its complexity, and many times one has to resort to dynamics-specific approaches that limit a comprehension of general principles.

To address this challenge, we employ a metric - that we name **Jacobian distance** - which captures the spatiotemporal spreading of perturbations, enabling us to uncover the latent geometry inherent in network-driven processes.

<p align="center">
  <img src="figures/jacobian_geometry.png" width="900"/>
</p>

## The Jacobian distance

Let us consider a networked dynamical system
$$\dot{x}_k = f_k(x_1, \ldots, x_N) \equiv f_k(\mathbf{x}),$$
where $x_k(t)$ is a variable representing the state of node $k = 1, \ldots, N$ at time $t$. The steady state $\mathbf{x}^{\*}$ of the system is given by $f_k(\mathbf{x}^{\*}) = 0 \ \forall k$. The time evolution of the perturbation on any node $k$ follows, then,
$$\delta \dot{x}_k(t) = f_k( \mathbf{x}^* + \delta \mathbf{x}(t)).$$

In vectorial notation, we have that $\delta \mathbf{x} \_{(i)}(0) = \delta x_i \mathbf{e} _{i}$, where $\mathbf{e}\_{i}$ is the unitary vector in the $i$-direction, and
$$\delta \dot{\mathbf{x}}\_{(i)} (t) = \mathrm{J} (\mathbf{x}^{\*}) \delta \mathbf{x}\_{(i)}(t), $$ where $\mathrm{J}(\mathbf{x}^{\*})$ is the Jacobian matrix evaluated at the steady state, which in general depends both on the specific functional form of the vector fields $\mathbf{f}$ and the topology. The general solution is given by
$$\delta \mathbf{x}\_{(i)}(t) = e^{\mathrm{J}(\mathbf{x}^*) t} \delta \mathbf{x}\_{(i)}(0).$$

The **Jacobian distance** is then defined as the temporal evolution of the difference between two perturbations of intensity $\delta x_i$ and $\delta x_j$ initially placed in nodes $i$ and $j$
```math
d_\tau(i,j) = || \delta \mathbf{x}_{(i)}(\tau) - \delta \mathbf{x}_{(j)}(\tau) || \\
=  || e^{\mathrm{J}(\mathbf{x}^*) \tau} [\delta x_i \mathbf{e}_{i} - \delta x_j \mathbf{e}_{j}]  ||
```

Since we are interested in unveiling the emergent patterns that are most persistent at the mesoscale, it is natural to average the distance matrices,
```math
    \overline{d}(i,j) = \frac{1}{\tau_{\text{max}}}\sum_{\tau=1}^{\tau_{\text{max}}} d_\tau(i,j),    
```
up to a certain cutoff that we fix $\tau_{\text{max}} \approx N$. In this way, emergent mesoscale patterns, if any, are highlighted.

## How to compute the Jacobian distance

``` python
### Import functions
from utils import distance
from utils import CommonFunctions as CF

### Get the network
N = 128
HMshape = [2,2,32]
avklist = [1,3,20]

mat = HM.HMRandomGraph(HMshape, avklist)

### Define the dynamical process
dynamic = 'Epidemics'

### Define the dynamical parameters
B = 1.
R = 0.05
params = [R, B]

### Get steady state
initial_state = np.random.random(N)
steady_state = CF.Numerical_Integration(mat, dynamic, initial_state, show=True, args=params)
        
### Compute jacobian
jacobian = CF.Jacobian(mat, dynamic, steady_state[-1], norm=True, args=params)
jacobians.append(jacobian)
    
### Compute jacobian distance at various tau
avg_jacobian_distance = distance.average_distance(jacobian, tmax=None, display=True, return_snapshot=True)
```

## Implemented dynamical processes

<div align="center">
  
Dynamics | $\partial_{\tau}x_i=$ |
| :--------: | :-------: |
Biochemical | $F -B x_i - R \sum_j A_{ij} x_i x_j$ |
Epidemics | $-B x_i + R \sum_j A_{ij} (1-x_i)x_j$ |
Mutualistic | $B x_i (1 - x_i) + R \sum_j A_{ij} x_i \frac{x_j^b}{1+x_j^b}$ |
Neuronal | $-B x_i + C \tanh x_i + R \sum_j A_{ij} \tanh x_j$ |
Noisy voter model | $A - B x_i + \frac{C}{k_i} \sum_j A_{ij} x_j$ |
Population | $-B x_i^{b} + R \sum_j A_{ij} x_j^a$ |
Regulatory | $-B x_i^a + R \sum_j A_{ij} \frac{x_j^h}{1+x_j^h}$ |
Synchronization | $\omega_i + R \sum_j A_{ij} \sin(x_j-x_i)$ |

</div>
