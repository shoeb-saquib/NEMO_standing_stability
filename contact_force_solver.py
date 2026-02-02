import osqp
import numpy as np
import scipy.sparse as sp

MU = 0.6

class ContactForceSolver:
    def __init__(self):
        n_vars = 12  # 6 for left foot, 6 for right foot
        n_eq = 6     # 3 linear + 3 angular acceleration

        # Equality rows (updated each step)
        self.A_eq = np.ones((n_eq, n_vars))
        self.l_eq = np.zeros(n_eq)
        self.u_eq = np.zeros(n_eq)

        # Friction constraints for both feet
        A_fric = sp.csc_matrix([
            # ---- LEFT FOOT ----
            [1,0,-MU, 0,0,0, 0,0,0, 0,0,0],
            [-1,0,-MU, 0,0,0, 0,0,0, 0,0,0],
            [0,1,-MU, 0,0,0, 0,0,0, 0,0,0],
            [0,-1,-MU, 0,0,0, 0,0,0, 0,0,0],
            [0,0, 1, 0,0,0, 0,0,0, 0,0,0],

            # ---- RIGHT FOOT ----
            [0,0,0, 0,0,0,  1,0,-MU, 0,0,0],
            [0,0,0, 0,0,0, -1,0,-MU, 0,0,0],
            [0,0,0, 0,0,0,  0,1,-MU, 0,0,0],
            [0,0,0, 0,0,0,  0,-1,-MU, 0,0,0],
            [0,0,0, 0,0,0,  0,0,1, 0,0,0],
        ])

        l_fric = np.array([-np.inf,-np.inf,-np.inf,-np.inf,0, -np.inf,-np.inf,-np.inf,-np.inf,0])
        u_fric = np.array([0,0,0,0,np.inf, 0,0,0,0,np.inf])

        # Stack equality + friction
        self.A_full = sp.vstack([self.A_eq, A_fric]).tocsc()
        self.l_full = np.hstack([self.l_eq, l_fric])
        self.u_full = np.hstack([self.u_eq, u_fric])

        P = sp.eye(n_vars, format='csc') * 2
        q = np.zeros(n_vars)

        self.osqp = osqp.OSQP()
        self.osqp.setup(P=P, q=q, A=self.A_full, l=self.l_full, u=self.u_full, warm_start=True, verbose=False)

    def solve(self, force_to_accel, desired_accel):

        self.A_full[:self.A_eq.shape[0], :] = force_to_accel
        self.l_full[:self.A_eq.shape[0]] = desired_accel
        self.u_full[:self.A_eq.shape[0]] = desired_accel


        self.osqp.update(Ax=self.A_full.data, l=self.l_full, u=self.u_full)
        result = self.osqp.solve()
        if result.x is None:
            return np.linalg.pinv(force_to_accel) @ desired_accel
        return result.x
