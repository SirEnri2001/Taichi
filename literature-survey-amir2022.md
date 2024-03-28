# Literature Survey for Compact Poisson Filter for Fast Fluid Simulation

https://docs.qq.com/doc/DTmZHRFlla0JJWlBX?_t=1706273458665&u=d73228a2951046ec93031e6959be4db8

https://dl.acm.org/doi/pdf/10.1145/3528233.3530737

Applied a new Possion filter-based solver. Derive universal Poisson kernels for forward and inverse Poisson problems.

• a theory of localized and factorized Poisson filter kernels,
• analytic derivations for compact realizations of these kernels,

Related methods

- Subspace and Spectral Methods.

- Operator Factorizations.

- Efficient Solvers.

- Multigrid.

- Learning Surrogates.

On Section 2, the author rely on spectral modes to accelerate primal domain simulation.

Poisson equations
$$\nabla^2\varphi = f$$
Inverse Poisson problem: $f$ is given and $\varphi$ is sought

Divergence of u
$$(\nabla \cdot \bold{u})_{i,j}=(\bold{u}_{i+1,j}^x-\bold{u}_{i,j}^x+\bold{u}_{i,j+1}^y-\bold{u}_{i,j}^y)$$

Laplacian operator of p
$$\nabla^2p_{i,j}=\frac{1}{\Delta x^2}(-4p_{i,j}+p_{i+1,j}+p_{i-1,j}+p_{i,j-1}+p_{i,j+1})$$

Chorin-style projection
$$\nabla^2 p = \frac{\rho}{\Delta t}\nabla \cdot \bold{u}$$

Code:
```python
@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25
```

