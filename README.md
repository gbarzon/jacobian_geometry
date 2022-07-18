# Jacobian geometry

## Changes

V parameter "perturbation_strength" is not needed when computing the jacobian\
V check if normalization is correct, i.e., total number of node couples -> N(N-1)/2\
V comparison diffusion distance - jacobian distance\
V create functions for plotting (instead of copy and paste)\
V show inter- vs intra- cluster distance\
V varying parameters in a grid\
V Implement hierarchical clustering\
V Show dendrogram beside average distance matrix\
V Compute Mantel test btw different average matrices\
V Simulating diffusion distance with different parameters\
    -> if normalized, it seems independent of A -> maybe A related to decay (indeed A do not depend on topology)\
V Return eigenvalues from diffusion/jacobian functions\
V Plot eigenvalues of jacobian\
V Initial trend -> alpha == lambda max == A\
  Final trend -> alpha == lambda min == A-B\
V Define a characteristic time -> lambda max\
V Plot everything with same cmap\
V Automatize for various parameters

## TODO