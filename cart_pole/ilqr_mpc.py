import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Callable

def _stars_bars(n: int, k: int):
    if k == 0:
        return [tuple([0]*n)]
    out = []
    def rec(prefix, rem, d):
        if d == 1:
            out.append(tuple(prefix + [rem]))
            return
        for v in range(rem+1):
            rec(prefix+[v], rem-v, d-1)
    rec([], k, n)
    return out

def build_exps(n: int, dmax: int) -> Dict[int, np.ndarray]:
    return {k: np.array(_stars_bars(n, k), dtype=int) for k in range(dmax+1)}

def monomials_and_grads(x: np.ndarray, exps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mon = prod_j x_j^{e_ij} for each monomial i,
    and dmon_dx (M,n) analytically.

    Robust at x_j == 0 using special-cases:
      - if e==0 => derivative 0
      - if x_j==0 and e==1 => derivative is product of other terms
      - if x_j==0 and e>=2 => derivative 0
    """
    x = np.asarray(x, dtype=float)
    exps = np.asarray(exps, dtype=int)
    M, n = exps.shape

    # monomials
    # (safe for 0^0 because exponent 0 -> treat as 1)
    mon = np.ones((M,), dtype=float)
    for j in range(n):
        ej = exps[:, j]
        if np.any(ej != 0):
            mon *= np.power(x[j], ej, where=(ej != 0), out=np.ones_like(mon))

    dmon_dx = np.zeros((M, n), dtype=float)

    # fast path for x_j != 0: dmon/dx_j = mon * e_ij / x_j  (when e_ij>0)
    for j in range(n):
        ej = exps[:, j]
        if abs(x[j]) > 0.0:
            mask = ej > 0
            if np.any(mask):
                dmon_dx[mask, j] = mon[mask] * ej[mask] / x[j]
        else:
            # x[j] == 0: handle exactly
            # d/dx (x^e) at 0:
            # e=0 -> 0, e=1 -> 1, e>=2 -> 0
            idx = np.where(ej == 1)[0]
            if idx.size > 0:
                # product of other dimensions for those monomials
                for i in idx:
                    prod_other = 1.0
                    for l in range(n):
                        if l == j:
                            continue
                        e_il = exps[i, l]
                        if e_il != 0:
                            prod_other *= x[l] ** e_il
                    dmon_dx[i, j] = prod_other

    return mon, dmon_dx


class PolyDynamicsWithJac:
    """
    Wrap your existing drift/inp dicts and exponents to provide:
      - continuous dynamics g(x,u) = f(x)+G(x)u
      - Jacobians gx = dg/dx, gu = dg/du
      - discrete step (Euler/RK4) + Jacobians A=dx_next/dx, B=dx_next/du
    """
    # def __init__(self, cfg, exps: Dict, drift: Dict, inp: Dict, eval_monomials: Callable):
    def __init__(self, cfg, drift: Dict, inp: Dict):
        self.cfg = cfg
        self.drift = drift
        self.inp = inp
        # self.eval_monomials = eval_monomials  # your function if you want; we won't rely on it

        self.n = int(cfg.n)
        self.m = int(cfg.m)
        self.dt = float(cfg.dt)
        self.integrator = cfg.integrator
        self.exps = build_exps(cfg.n, max(cfg.d_f, cfg.d_G))

        # self.exps = build_exps(cfg.n, max(cfg.d_f, cfg.d_G)),
    def eval_f_and_jac(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        drift = self.drift
        x = np.asarray(x, dtype=float)

        y = drift["f0"] + drift["A"] @ x
        J = drift["A"].copy()  # (n,n)

        for k, Ck in drift["C"].items():
            if Ck.size == 0:
                continue
            exps_k = self.exps[k]              # (M_k,n)
            mon, dmon_dx = monomials_and_grads(x, exps_k)  # (M_k,), (M_k,n)
            y = y + mon @ Ck                   # (n,)
            # Jacobian contribution: (dmon_dx)^T @ Ck  -> (n,n)
            J = J + dmon_dx.T @ Ck

        return y, J

    def eval_G_and_jac(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          G(x): (n,m)
          dGdx: (n, n, m) where dGdx[:, j, :] = ∂G/∂x_j  (n,m)
        """
        inp = self.inp
        x = np.asarray(x, dtype=float)

        G = inp["G0"].copy()  # (n,m)
        dGdx = np.zeros((self.n, self.n, self.m), dtype=float)

        for k, Dk in inp["D"].items():
            if Dk.size == 0:
                continue
            exps_k = self.exps[k]  # (M_k,n)
            mon, dmon_dx = monomials_and_grads(x, exps_k)  # (M_k,), (M_k,n)

            # G += sum_i mon_i * Dk[i,:,:]
            G = G + np.tensordot(mon, Dk, axes=(0, 0))  # (n,m)

            # For each state dimension j:
            # ∂G/∂x_j = sum_i (∂mon_i/∂x_j) * Dk[i,:,:]
            for j in range(self.n):
                w = dmon_dx[:, j]  # (M_k,)
                if np.any(w != 0.0):
                    dGdx[:, j, :] += np.tensordot(w, Dk, axes=(0, 0))  # (n,m)

        return G, dGdx

    def g_and_jac(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Continuous dynamics:
          g(x,u) = f(x) + G(x) u

        Returns:
          g: (n,)
          gx = ∂g/∂x : (n,n)
          gu = ∂g/∂u : (n,m) = G(x)
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        f, dfdx = self.eval_f_and_jac(x)      # (n,), (n,n)
        G, dGdx = self.eval_G_and_jac(x)      # (n,m), (n,n,m)

        g = f + G @ u                         # (n,)
        gu = G                                # (n,m)

        # gx[:, j] = dfdx[:, j] + (∂G/∂x_j) @ u
        gx = dfdx.copy()
        for j in range(self.n):
            gx[:, j] += dGdx[:, j, :] @ u

        return g, gx, gu

    def step_and_jac(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Discrete step x_{k+1} = step(x_k,u_k) and Jacobians:
          A = ∂x_next/∂x,  B = ∂x_next/∂u
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        n, m, dt = self.n, self.m, self.dt

        I = np.eye(n)

        if self.integrator == "euler":
            k1, gx1, gu1 = self.g_and_jac(x, u)
            x_next = x + dt * k1
            A = I + dt * gx1
            B = dt * gu1
            return x_next, A, B

        if self.integrator == "rk4":
            # k1
            k1, gx1, gu1 = self.g_and_jac(x, u)

            # x2 = x + 0.5 dt k1
            x2 = x + 0.5 * dt * k1
            dx2_dx = I + 0.5 * dt * gx1
            dx2_du = 0.5 * dt * gu1

            # k2
            k2, gx2, gu2 = self.g_and_jac(x2, u)
            dk2_dx = gx2 @ dx2_dx
            dk2_du = gx2 @ dx2_du + gu2

            # x3 = x + 0.5 dt k2
            x3 = x + 0.5 * dt * k2
            dx3_dx = I + 0.5 * dt * dk2_dx
            dx3_du = 0.5 * dt * dk2_du

            # k3
            k3, gx3, gu3 = self.g_and_jac(x3, u)
            dk3_dx = gx3 @ dx3_dx
            dk3_du = gx3 @ dx3_du + gu3

            # x4 = x + dt k3
            x4 = x + dt * k3
            dx4_dx = I + dt * dk3_dx
            dx4_du = dt * dk3_du

            # k4
            k4, gx4, gu4 = self.g_and_jac(x4, u)
            dk4_dx = gx4 @ dx4_dx
            dk4_du = gx4 @ dx4_du + gu4

            x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            A = I + (dt / 6.0) * (gx1 + 2*dk2_dx + 2*dk3_dx + dk4_dx)
            B = (dt / 6.0) * (gu1 + 2*dk2_du + 2*dk3_du + dk4_du)

            return x_next, A, B

        raise ValueError("integrator must be 'euler' or 'rk4'")

@dataclass
class ILQRMPCConfig:
    H: int = 25
    max_iters: int = 15
    tol_cost: float = 1e-8
    reg_init: float = 1e-3
    reg_factor: float = 10.0
    reg_min: float = 1e-8
    reg_max: float = 1e6
    alphas: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)

    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    shift_fill: str = "zero"


def _as_bound_vec(v: Optional[np.ndarray], m: int) -> Optional[np.ndarray]:
    if v is None:
        return None
    v = np.asarray(v, dtype=float)
    if v.ndim == 0:
        return np.full((m,), float(v), dtype=float)
    if v.shape == (m,):
        return v
    raise ValueError("u_min/u_max must be scalar or shape (m,)")


from dataclasses import dataclass

@dataclass
class QuadCostConfig:
    x_goal: np.ndarray
    u_max: float
    r_safe: float

    W_S_TRACK: float = 1.0
    W_S_SAFE: float = 5.0
    W_S_U: float = 0.1
    W_S_SMOOTH: float = 0.1

    W_T_TRACK: float = 10.0
    W_T_SAFE: float = 20.0
    W_T_U: float = 0.1
    W_T_SMOOTH: float = 0.0


class FastQuadraticCost:
    """
    Provides stage/terminal cost and derivatives for iLQR.

    Uses safe penalty: sum_i max(0, |x_i|-R)^2
    """
    def __init__(self, cfg: QuadCostConfig):
        self.cfg = cfg
        self.x_goal = np.asarray(cfg.x_goal, dtype=float)

    def _weights(self, terminal: bool):
        c = self.cfg
        if terminal:
            return c.W_T_TRACK, c.W_T_SAFE, c.W_T_U, c.W_T_SMOOTH
        return c.W_S_TRACK, c.W_S_SAFE, c.W_S_U, c.W_S_SMOOTH

    def l_and_derivs(
        self,
        x: np.ndarray,
        u: np.ndarray,
        u_prev: np.ndarray,
        terminal: bool,
    ):
        """
        Returns:
          l (scalar),
          lx (n,), lu (m,),
          lxx (n,n), luu (m,m), lux (m,n)
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        u_prev = np.asarray(u_prev, dtype=float)

        w_track, w_safe, w_u, w_smooth = self._weights(terminal)

        # tracking
        e = x - self.x_goal
        l_track = w_track * (e @ e)
        lx_track = 2.0 * w_track * e
        lxx_track = 2.0 * w_track * np.eye(x.size)

        # control magnitude (normalized)
        inv_um2 = 1.0 / (self.cfg.u_max ** 2 + 1e-12)
        l_u = w_u * (u @ u) * inv_um2
        lu_u = 2.0 * w_u * u * inv_um2
        luu_u = 2.0 * w_u * inv_um2 * np.eye(u.size)

        # smoothness
        du = u - u_prev
        l_sm = w_smooth * (du @ du)
        lu_sm = 2.0 * w_smooth * du
        luu_sm = 2.0 * w_smooth * np.eye(u.size)

        # safe: sum_i max(0, |x_i|-R)^2
        R = self.cfg.r_safe
        ax = np.abs(x)
        t = ax - R
        pos = np.maximum(0.0, t)
        l_safe = w_safe * np.sum(pos * pos)

        # gradient wrt x
        # d/dx_i: 2*pos_i*sign(x_i) if |x_i|>R else 0
        sgn = np.sign(x)
        active = t > 0.0
        lx_safe = np.zeros_like(x)
        lx_safe[active] = 2.0 * w_safe * pos[active] * sgn[active]

        # Hessian wrt x is piecewise constant: 2*w_safe on active dims (ignoring kink at 0)
        lxx_safe = np.zeros((x.size, x.size), dtype=float)
        for i in range(x.size):
            if active[i]:
                lxx_safe[i, i] = 2.0 * w_safe

        # assemble
        l = float(l_track + l_safe + l_u + l_sm)
        lx = lx_track + lx_safe
        lu = lu_u + lu_sm

        lxx = lxx_track + lxx_safe
        luu = luu_u + luu_sm
        lux = np.zeros((u.size, x.size), dtype=float)

        return l, lx, lu, lxx, luu, lux


class FastILQRMPC:
    """
    iLQR MPC with analytic dynamics Jacobians from PolyDynamicsWithJac,
    and analytic quadratic-ish cost derivatives from FastQuadraticCost.

    Augmented state z = [x; u_prev] to support smoothness term.
    """
    def __init__(
        self,
        dyn: PolyDynamicsWithJac,
        cost: FastQuadraticCost,
        cfg: ILQRMPCConfig,
        u_init: Optional[np.ndarray] = None,
        u_prev_init: Optional[np.ndarray] = None,
    ):
        self.dyn = dyn
        self.cost = cost
        self.cfg = cfg

        self.n = dyn.n
        self.m = dyn.m
        self.nz = self.n + self.m

        H = int(cfg.H)
        if u_init is None:
            self.U = np.zeros((H, self.m), dtype=float)
        else:
            u_init = np.asarray(u_init, dtype=float)
            assert u_init.shape == (H, self.m)
            self.U = u_init.copy()

        self.u_prev_applied = np.zeros((self.m,), dtype=float) if u_prev_init is None else np.asarray(u_prev_init, dtype=float)

        self.u_min = _as_bound_vec(cfg.u_min, self.m)
        self.u_max = _as_bound_vec(cfg.u_max, self.m)

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        if self.u_min is None and self.u_max is None:
            return u
        lo = self.u_min if self.u_min is not None else -np.inf
        hi = self.u_max if self.u_max is not None else np.inf
        return np.clip(u, lo, hi)

    def _f_aug(self, z: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Augmented dynamics:
          x_next = step(x,u)
          u_prev_next = u

        Returns z_next, A=dz_next/dz, B=dz_next/du
        """
        x = z[: self.n]
        # u_prev = z[self.n:] is used only for cost

        x_next, A_x, B_x = self.dyn.step_and_jac(x, u)

        z_next = np.zeros((self.nz,), dtype=float)
        z_next[: self.n] = x_next
        z_next[self.n :] = u

        A = np.zeros((self.nz, self.nz), dtype=float)
        B = np.zeros((self.nz, self.m), dtype=float)

        # dz_next/dx
        A[: self.n, : self.n] = A_x
        # dz_next/du_prev = 0 for x_next
        # u_prev_next = u => derivative wrt u goes to bottom block
        B[: self.n, :] = B_x
        B[self.n :, :] = np.eye(self.m)

        # derivative of u_prev_next wrt z is zero
        return z_next, A, B

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> Tuple[float, np.ndarray]:
        H = self.cfg.H
        U = np.asarray(U, dtype=float)
        x0 = np.asarray(x0, dtype=float)

        z = np.zeros((self.nz,), dtype=float)
        z[: self.n] = x0
        z[self.n :] = self.u_prev_applied

        Z = np.zeros((H + 1, self.nz), dtype=float)
        Z[0] = z

        J = 0.0
        for k in range(H):
            u = self._clip_u(U[k])
            x = z[: self.n]
            u_prev = z[self.n :]
            terminal = (k == H - 1)
            l, *_ = self.cost.l_and_derivs(x, u, u_prev, terminal)
            J += l
            z, _, _ = self._f_aug(z, u)
            Z[k + 1] = z

        return float(J), Z

    def optimize(self, x0: np.ndarray) -> Dict[str, Any]:
        H, m, nz = self.cfg.H, self.m, self.nz
        U = self._clip_u(self.U.copy())

        J, Z = self.rollout(x0, U)
        reg = float(self.cfg.reg_init)

        for it in range(self.cfg.max_iters):
            # collect linearizations + cost quadratics along trajectory
            A_list = []
            B_list = []
            lz_list = []
            lu_list = []
            lzz_list = []
            luu_list = []
            luz_list = []

            for k in range(H):
                z = Z[k]
                x = z[: self.n]
                u_prev = z[self.n :]
                u = self._clip_u(U[k])
                terminal = (k == H - 1)

                # cost derivatives w.r.t (x,u); map to (z,u)
                l, lx, lu, lxx, luu, lux = self.cost.l_and_derivs(x, u, u_prev, terminal)

                # lift to augmented z=[x; u_prev]
                lz = np.zeros((nz,), dtype=float)
                lz[: self.n] = lx
                # derivative wrt u_prev comes only from smoothness term, already inside lu above (since cost uses u-u_prev)
                # but for iLQR we need l(z,u) derivatives: include d l / d u_prev in lz:
                # smoothness term = w*||u-u_prev||^2 => d/du_prev = -2w(u-u_prev)
                # Our FastQuadraticCost doesn't return it explicitly, so compute here consistently:
                # We can infer it by reusing same weights:
                # easiest: finite compute from formula (cheap):
                # We'll replicate smoothness derivative from cost config:
                c = self.cost.cfg
                w_track, w_safe, w_u, w_smooth = (c.W_T_TRACK, c.W_T_SAFE, c.W_T_U, c.W_T_SMOOTH) if terminal else (c.W_S_TRACK, c.W_S_SAFE, c.W_S_U, c.W_S_SMOOTH)
                lz[self.n:] = -2.0 * w_smooth * (u - u_prev)

                lzz = np.zeros((nz, nz), dtype=float)
                lzz[: self.n, : self.n] = lxx
                # d^2/d(u_prev)^2 for smoothness: 2*w_smooth*I
                lzz[self.n:, self.n:] = 2.0 * w_smooth * np.eye(m)
                # cross between x and u_prev is zero here

                # cost Hessian cross terms between u and z: luz is (m,nz)
                # We have lux = d^2 l / du dx, and cross with u_prev from smoothness: d^2 l / du d(u_prev) = -2*w_smooth*I
                luz_full = np.zeros((m, nz), dtype=float)
                luz_full[:, : self.n] = lux
                luz_full[:, self.n:] = -2.0 * w_smooth * np.eye(m)

                # dynamics linearization
                z_next, A, B = self._f_aug(z, u)

                A_list.append(A)
                B_list.append(B)
                lz_list.append(lz)
                lu_list.append(lu)
                lzz_list.append(lzz)
                luu_list.append(luu)
                luz_list.append(luz_full)

            # backward pass
            Vz = np.zeros((nz,), dtype=float)
            Vzz = np.zeros((nz, nz), dtype=float)

            k_ff = np.zeros((H, m), dtype=float)
            K_fb = np.zeros((H, m, nz), dtype=float)

            success = True
            for k in reversed(range(H)):
                A = A_list[k]
                B = B_list[k]
                lz = lz_list[k]
                lu = lu_list[k]
                lzz = lzz_list[k]
                luu = luu_list[k]
                luz = luz_list[k]

                Qz = lz + A.T @ Vz
                Qu = lu + B.T @ Vz

                Qzz = lzz + A.T @ Vzz @ A
                Quu = luu + B.T @ Vzz @ B
                Quz = luz + B.T @ Vzz @ A

                Quu_reg = Quu + reg * np.eye(m)

                try:
                    L = np.linalg.cholesky(Quu_reg)
                    def solve(b):
                        y = np.linalg.solve(L, b)
                        return np.linalg.solve(L.T, y)
                except np.linalg.LinAlgError:
                    success = False
                    break

                k_k = -solve(Qu)
                K_k = -solve(Quz)

                k_ff[k] = k_k
                K_fb[k] = K_k

                Vz = Qz + K_k.T @ Quu @ k_k + K_k.T @ Qu + Quz.T @ k_k
                Vzz = Qzz + K_k.T @ Quu @ K_k + K_k.T @ Quz + Quz.T @ K_k
                Vzz = 0.5 * (Vzz + Vzz.T)

            if not success:
                reg *= self.cfg.reg_factor
                if reg > self.cfg.reg_max:
                    break
                continue

            # forward line search
            improved = False
            best_J = J
            best_U = U

            for a in self.cfg.alphas:
                U_try = np.zeros_like(U)

                # initial augmented state
                z = np.zeros((nz,), dtype=float)
                z[: self.n] = np.asarray(x0, dtype=float)
                z[self.n :] = self.u_prev_applied

                J_try = 0.0
                for k in range(H):
                    dz = z - Z[k]
                    u = self._clip_u(U[k] + a * k_ff[k] + K_fb[k] @ dz)
                    U_try[k] = u

                    x = z[: self.n]
                    u_prev = z[self.n :]
                    terminal = (k == H - 1)
                    l, *_ = self.cost.l_and_derivs(x, u, u_prev, terminal)
                    J_try += l

                    z, _, _ = self._f_aug(z, u)

                if J_try < best_J - self.cfg.tol_cost:
                    best_J = J_try
                    best_U = U_try
                    improved = True
                    break

            if improved:
                if abs(J - best_J) < self.cfg.tol_cost:
                    U = best_U
                    J = best_J
                    break
                U = best_U
                J = best_J
                reg = max(self.cfg.reg_min, reg / self.cfg.reg_factor)
                # refresh rollout for next iter
                _, Z = self.rollout(x0, U)
            else:
                reg *= self.cfg.reg_factor
                if reg > self.cfg.reg_max:
                    break

        self.U = U
        return {"status": "ok", "J": float(J), "U": self.U.copy()}

    def act(self, x: np.ndarray) -> np.ndarray:
        self.optimize(x)
        return self.U[0].copy()

    def shift_after_apply(self, u_applied: np.ndarray):
        u_applied = np.asarray(u_applied, dtype=float)
        self.u_prev_applied = u_applied.copy()

        if self.cfg.H <= 1:
            return
        if self.cfg.shift_fill == "repeat_last":
            last = self.U[-1].copy()
        else:
            last = np.zeros((self.m,), dtype=float)
        self.U[:-1] = self.U[1:]
        self.U[-1] = last


def run_closed_loop_fast_ilqr_mpc(
    mpc: FastILQRMPC,
    step_fn_true: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0: np.ndarray,
    T_outer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x0, dtype=float).copy()
    n = x.shape[0]
    m = mpc.m

    X_cl = np.zeros((T_outer + 1, n), dtype=float)
    U_cl = np.zeros((T_outer, m), dtype=float)
    X_cl[0] = x

    for t in range(T_outer):
        u = mpc.act(x)
        U_cl[t] = u
        x = step_fn_true(x, u)   # use your exact step for closed-loop sim
        X_cl[t + 1] = x
        mpc.shift_after_apply(u)

    return X_cl, U_cl