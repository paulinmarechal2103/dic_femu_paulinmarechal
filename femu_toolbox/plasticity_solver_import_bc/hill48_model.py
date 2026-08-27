"""
hill48_model.py
---------------
Hill48 orthotropic anisotropic plasticity model with Voce isotropic hardening.

Formulation
~~~~~~~~~~~
Equivalent Stress:
  σ_hill48(σ) = √[ F(σ₂₂ - σ₃₃)² + G(σ₃₃ - σ₁₁)² + H(σ₁₁ - σ₂₂)² 
                 + 2L σ₂₃² + 2M σ₁₃² + 2N σ₁₂² ]

Yield Function:
  f(σ, p) = σ_hill48(σ) – σ_Y – Q · (1 – exp(–k · p))

Plastic Flow Rule:
  Δεᵖ = Δp · n,   where n = ∂σ_hill48 / ∂σ
"""

import ufl
from dolfinx import fem

from .simu_tools import ElasticModel, PlasticState, PlasticityModel


class Hill48state(PlasticState):
    """
    Internal history variable storage for the Hill48 anisotropic model.

    Attributes
    ----------
    p_old : fem.Function (scalar DG-0)
        Accumulated equivalent plastic strain p at step tₙ.
    eps_p_old : fem.Function (tensor DG-0)
        Plastic strain tensor εᵖ at step tₙ.
    """

    def __init__(self, W, WT):
        self.p_old     = fem.Function(W)   # cumulative plastic strain p
        self.eps_p_old = fem.Function(WT)  # plastic strain tensor εᵖ
        self._W        = W
        self._WT       = WT


class Hill48Model(PlasticityModel):
    """
    Hill48 anisotropic plasticity model with Voce isotropic hardening.

    Parameters
    ----------
    elastic : ElasticModel
        Shared linear-elastic model instance.
    sigma_Y : float
        Initial yield stress [MPa].
    F, G, H : float
        Hill48 normal anisotropy parameters along orthogonal material axes.
    L, M, N : float
        Hill48 shear anisotropy parameters along shear planes.
    Q_var : float
        Voce isotropic hardening saturation stress [MPa].
    k_hardening : float
        Voce hardening rate exponent parameter.
    """

    def __init__(
        self,
        elastic: ElasticModel,
        sigma_Y: float,
        H: float,
        F: float,
        G: float,
        L: float,
        M: float,
        N: float,
        Q_var: float,
        k_hardening: float,
    ):
        super().__init__(elastic)
        self.sigma_Y = sigma_Y
        self.H = H
        self.F = F
        self.G = G
        self.L = L
        self.M = M
        self.N = N
        self.Q_var = Q_var
        self.k = k_hardening

    # -- Internal UFL helper methods ---------------------------------------

    def _sigma_hill48(self, sigma):
        """
        Compute Hill48 equivalent stress UFL expression.
        
        Includes a small numerical regulariser (+ 1e-12) to prevent zero division
        when evaluating ∂σ_hill48/∂σ in the elastic regime.
        """
        return ufl.sqrt(
            self.F * (sigma[1, 1] - sigma[2, 2]) ** 2
            + self.G * (sigma[2, 2] - sigma[0, 0]) ** 2
            + self.H * (sigma[0, 0] - sigma[1, 1]) ** 2
            + 2.0 * self.L * sigma[1, 2] ** 2
            + 2.0 * self.M * sigma[0, 2] ** 2
            + 2.0 * self.N * sigma[0, 1] ** 2
            + 1e-12
        )

    def _yield_func(self, sigma, p):
        """Yield function f = σ_hill48(σ) - σ_Y - Q*(1 - exp(-k*p))."""
        R = self.Q_var * (1.0 - ufl.exp(-self.k * p))
        return self._sigma_hill48(sigma) - self.sigma_Y - R

    def _flow_normal(self, sigma):
        """
        Compute normalized plastic flow direction n = ∂σ_hill48 / ∂σ (associated flow rule).
        """
        s_hill = self._sigma_hill48(sigma) + 1e-10
        
        # Derivatives of Hill48 quadratic form with respect to stress components
        dPhi_00 = 2.0 * self.G * (sigma[0, 0] - sigma[2, 2]) + 2.0 * self.H * (sigma[0, 0] - sigma[1, 1])
        dPhi_11 = 2.0 * self.F * (sigma[1, 1] - sigma[2, 2]) + 2.0 * self.H * (sigma[1, 1] - sigma[0, 0])
        dPhi_22 = 2.0 * self.F * (sigma[2, 2] - sigma[1, 1]) + 2.0 * self.G * (sigma[2, 2] - sigma[0, 0])

        dPhi_01 = 2.0 * self.N * sigma[0, 1]
        dPhi_10 = 2.0 * self.N * sigma[1, 0]
        dPhi_02 = 2.0 * self.M * sigma[0, 2]
        dPhi_20 = 2.0 * self.M * sigma[2, 0]
        dPhi_12 = 2.0 * self.L * sigma[1, 2]
        dPhi_21 = 2.0 * self.L * sigma[2, 1]

        dPhi = ufl.as_tensor([
            [dPhi_00, dPhi_01, dPhi_02],
            [dPhi_10, dPhi_11, dPhi_12],
            [dPhi_20, dPhi_21, dPhi_22],
        ])

        return (1.0 / (2.0 * s_hill)) * dPhi

    # -- PlasticityModel interface -----------------------------------------

    def create_state(self, domain, W, WT) -> Hill48state:
        return Hill48state(W, WT)

    def update(self, state: Hill48state, eps) -> tuple:
        """
        Linearised radial return-mapping step for Hill48 orthotropic plasticity.
        """
        # 1. Elastic trial predictor
        sigma_tr = self.elastic.sigma(eps - state.eps_p_old)
        f_val    = self._yield_func(sigma_tr, state.p_old)
        n        = self._flow_normal(sigma_tr)

        # 2. Hardening law derivative: dR/dp = Q * k * exp(-k * p_old)
        R_prime  = self.Q_var * self.k * ufl.exp(-self.k * state.p_old)

        # 3. Anisotropic plastic modulus projection E_n = n : C : n
        E_n     = ufl.inner(n, self.elastic.sigma(n))
        f_prime = -R_prime - E_n

        # 4. Plastic multiplier update
        delta_p0 = -(1.0 / f_prime) * f_val

        # 5. UFL conditional plastic step
        delta_eps_p = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0 * n, 0.0 * n)
        delta_p     = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0, 0.0)
        return delta_p, delta_eps_p

    def commit(self, state: Hill48state, uh) -> None:
        """
        Advance Hill48 history variables (p and εᵖ) in-place at step completion.
        """
        eps                  = self.elastic.epsilon(uh)
        delta_p, delta_eps_p = self.update(state, eps)

        state.p_old.interpolate(
            fem.Expression(state.p_old + delta_p, state._W.element.interpolation_points)
        )
        delta_eps_p_proj = fem.Function(state._WT)
        delta_eps_p_proj.interpolate(
            fem.Expression(delta_eps_p, state._WT.element.interpolation_points)
        )
        state.eps_p_old.x.array[:] += delta_eps_p_proj.x.array[:]
