from casadi import *
import pendulum_model
from casadi.tools import struct_symSX, entry
from scipy import constants
import numpy as np


class mpc_controller(object):
    def __init__(self, Q, R, P, dt, N, mpc_steps, K, x0, xz_pos, xz_ori,
                 circles):
        self.dt = dt  #timestep
        self.N = N  #prediction horizon
        self.mpc_steps = mpc_steps
        self.K = K  #collocation degree

        self.x0 = x0
        self.xz_pos = xz_pos  #target position
        self.xz_ori = xz_ori  #target orientation

        self.circles = circles

        self.nx = 6
        self.nu = 1

        self.stage_cost_fcn, self.terminal_cost_fcn = self._defineCostFunctions(
            pendulum_model.x, pendulum_model.u, Q, R, P)

        self.opt_x, self.lb_opt_x, self.ub_opt_x = self._defineOptimizationVariablesAndBounds(
        )

        self.A, self.D = self._prepareCoefficients()

        opts = {'tf': self.dt}

        # Create the solver object
        self.ode_solver = integrator('F', 'idas', pendulum_model.ode, opts)

        self.lb_g, self.ub_g, self.mpc_solver = self._formulateOptimizationProblem(
        )

    def _defineCostFunctions(self, x, u, Q, R, P):

        T_ges = pendulum_model.T_ges
        V_ges = pendulum_model.V_ges

        if self.xz_ori == 'up_Dev':
            stage_cost = Q[0] * (x[0] - self.xz_pos)**2 + Q[1] * sin(
                0.5 * (x[1] - 0))**2 + Q[2] * sin(0.5 *
                                                  (x[2] - 0))**2 + R * u**2

            terminal_cost = P[0] * (x[0] - self.xz_pos)**2 + P[1] * sin(
                0.5 * (x[1] - 0))**2 + P[2] * sin(
                    0.5 * (x[2] - 0))**2 + P[3] * (x[3]**2 + x[4]**2 + x[5]**2)

            stage_cost_fcn = Function('stage_cost', [x, u], [stage_cost])
            terminal_cost_fcn = Function('terminal_cost', [x], [terminal_cost])

        if self.xz_ori == 'down_Dev':
            stage_cost = Q[0] * (x[0] - self.xz_pos)**2 + Q[1] * sin(
                0.5 *
                (x[1] - np.pi))**2 + Q[2] * sin(0.5 *
                                                (x[2] - np.pi))**2 + R * u**2

            terminal_cost = P[0] * (x[0] - self.xz_pos)**2 + P[1] * sin(
                0.5 * (x[1] - np.pi))**2 + P[2] * sin(
                    0.5 *
                    (x[2] - np.pi))**2 + P[3] * (x[3]**2 + x[4]**2 + x[5]**2)

            stage_cost_fcn = Function('stage_cost', [x, u], [stage_cost])
            terminal_cost_fcn = Function('terminal_cost', [x], [terminal_cost])

        if self.xz_ori == 'up_E':

            stage_cost = Q[0] * T_ges - Q[1] * V_ges + Q[2] * (
                x[0] - self.xz_pos)**2 + u**2 * R

            terminal_cost = P[0] * T_ges - P[1] * V_ges + P[2] * (
                x[0] - self.xz_pos)**2

            stage_cost_fcn = Function('stage_cost', [x, u], [stage_cost])
            terminal_cost_fcn = Function('terminal_cost', [x], [terminal_cost])

        if self.xz_ori == 'down_E':

            stage_cost = Q[0] * T_ges + Q[1] * V_ges + Q[2] * (
                x[0] - self.xz_pos)**2 + u**2 * R

            terminal_cost = P[0] * T_ges + P[1] * V_ges + P[2] * (
                x[0] - self.xz_pos)**2

            stage_cost_fcn = Function('stage_cost', [x, u], [stage_cost])
            terminal_cost_fcn = Function('terminal_cost', [x], [terminal_cost])

        return stage_cost_fcn, terminal_cost_fcn

    def _defineOptimizationVariablesAndBounds(self):

        # definition of constraints on x and u

        lb_x = 20 * np.array([-3, -2, -2, -2, -2, -2])
        ub_x = 20 * np.array([3, 2, 2, 2, 2, 2])

        lb_u = -4 * np.ones((self.nu, 1))
        ub_u = 4 * np.ones((self.nu, 1))

        opt_x = struct_symSX([
            entry('x', shape=self.nx, repeat=[self.N + 1, self.K + 1]),
            entry('u', shape=self.nu, repeat=[self.N])
        ])

        # initialize bounds with all zero according to opt_x structure
        lb_opt_x = opt_x(0)
        ub_opt_x = opt_x(0)

        # set bounds
        lb_opt_x['x'] = lb_x
        ub_opt_x['x'] = ub_x
        lb_opt_x['u'] = lb_u
        ub_opt_x['u'] = ub_u

        return opt_x, lb_opt_x, ub_opt_x

    def _getCircleConstraints(self, x_cart, theta1, theta2):
        # use more point on rods for higher constaint observance

        # pendulum state in cartesian coordinates
        p1x = pendulum_model.l1 * sin(theta1) + x_cart
        p1y = pendulum_model.l1 * cos(theta1)
        p2x = pendulum_model.l2 * sin(theta2) + p1x
        p2y = pendulum_model.l2 * cos(theta2) + p1y

        g = []
        circle_lb_g = []
        circle_ub_g = []

        for circle in self.circles:
            # constraint is distance from circle center to rod points, bound is radius
            constraint1 = (p1x - circle['center'][0])**2 + (
                p1y - circle['center'][1])**2
            constraint2 = (p2x - circle['center'][0])**2 + (
                p2y - circle['center'][1])**2
            # append to g, lbg and ubg
            g.append(constraint1)
            circle_lb_g.append(circle['radius']**2)
            circle_ub_g.append(np.inf)
            g.append(constraint2)
            circle_lb_g.append(circle['radius']**2)
            circle_ub_g.append(np.inf)

        return g, circle_lb_g, circle_ub_g

    def _prepareCoefficients(self):

        # collocation points including 0
        tau_col = collocation_points(self.K, 'legendre')
        tau_col = [0] + tau_col

        self.tau_col = tau_col

        # OC coefficents a_jk stored in A
        tau = SX.sym('tau')
        A = np.zeros((self.K + 1, self.K + 1))

        def L(tau_col, tau, j):
            l = 1
            for k in range(len(tau_col)):
                if k != j:
                    l *= (tau - tau_col[k]) / (tau_col[j] - tau_col[k])
            return l

        for j in range(self.K + 1):
            dLj = gradient(L(tau_col, tau, j), tau)
            dLj_fcn = Function('dLj_fcn', [tau], [dLj])
            for k in range(self.K + 1):
                A[j, k] = dLj_fcn(tau_col[k])

        # Continuity coefficients d_j stored in D
        D = np.zeros((self.K + 1, 1))

        for j in range(self.K + 1):
            Lj = L(tau_col, tau, j)
            Lj_fcn = Function('Lj', [tau], [Lj])
            D[j] = Lj_fcn(1)

        return A, D

    def _formulateOptimizationProblem(self):

        # Initialize empty list of constraints and Costs
        J = 0  #cost function
        g = []  # constraint expression g
        lb_g = []  # lower bound for constraint expression g
        ub_g = []  # upper bound for constraint expression g

        # introduce x_init as equality constrain for the first state
        x_init = SX.sym('x_init', self.nx)
        x_0 = self.opt_x['x', 0, 0]
        g.append(x_0 - x_init)

        lb_g.append(np.zeros((self.nx, 1)))
        ub_g.append(np.zeros((self.nx, 1)))

        for i in range(self.N):
            # Add stage cost (only for first collocation point)
            J += self.stage_cost_fcn(self.opt_x['x', i, 0], self.opt_x['u', i])

            # collocation constraints
            # equality constraints (system equation)
            for k in range(1, self.K + 1):
                gk = -self.dt * pendulum_model.systemdynamics(
                    self.opt_x['x', i, k], self.opt_x['u', i])
                for j in range(self.K + 1):
                    gk += self.A[j, k] * self.opt_x['x', i, j]

                g.append(gk)
                lb_g.append(np.zeros((self.nx, 1)))
                ub_g.append(np.zeros((self.nx, 1)))

            # add continuity contraints
            x_next = horzcat(*self.opt_x['x', i]) @ self.D
            g.append(x_next - self.opt_x['x', i + 1, 0])
            lb_g.append(np.zeros((self.nx, 1)))
            ub_g.append(np.zeros((self.nx, 1)))

            # append circle constraints
            circle_g, circle_glb, circle_gub = self._getCircleConstraints(
                self.opt_x['x', i, 0][0], self.opt_x['x', i, 0][1],
                self.opt_x['x', i, 0][2])
            circle_g = vertcat(*circle_g)
            circle_glb = vertcat(*circle_glb)
            circle_gub = vertcat(*circle_gub)
            g.append(circle_g)
            lb_g.append(circle_glb)
            ub_g.append(circle_gub)

        # Add terminal costs
        J += self.terminal_cost_fcn(self.opt_x['x', self.N, 0])

        # 06 - create nlpsol object and concatenate g and bounds
        g = vertcat(*g)
        lb_g = vertcat(*lb_g)
        ub_g = vertcat(*ub_g)
        nlpsol_opts = {
            #'ipopt.linear_solver': 'MA27'
        }  #'ipopt.linear_solver': 'mumps'
        prob = {'f': J, 'x': vertcat(self.opt_x), 'g': g, 'p': x_init}
        mpc_solver = nlpsol('solver', 'ipopt', prob, nlpsol_opts)

        return lb_g, ub_g, mpc_solver

    def mpc_firstprediction(self):

        x_0 = self.x0
        initial_guess = self.opt_x(0)
        initial_guess['x'] = self.x0

        lb_g, ub_g, mpc_solver = self._formulateOptimizationProblem()

        mpc_res = mpc_solver(p=x_0,
                             x0=initial_guess,
                             lbg=lb_g,
                             ubg=ub_g,
                             lbx=self.lb_opt_x,
                             ubx=self.ub_opt_x)

        opt_x_k = self.opt_x(mpc_res['x'])
        X_k = horzcat(*opt_x_k['x', :, 0, :])
        U_k = horzcat(*opt_x_k['u', :])

        cost_res = []
        for i in range(self.N - 1):
            cost_res.append(self.stage_cost_fcn(X_k[:, i], U_k[:, i]))

        cost_res = np.concatenate(cost_res, axis=1)

        fake_predictions = np.zeros((self.N + 1, 7, 1))

        return X_k, U_k, cost_res[0], fake_predictions

    def _simulateWithOC(
            self, x, u):  #ode-solver for simulation with othogonal collocation

        A = self.A
        D = self.D
        tau_col = self.tau_col

        #Create an optimization problem to solve the collocation problem
        g = []

        # States at all collocation points
        X = SX.sym('X', self.nx, self.K + 1)
        # Initial state
        x_init = SX.sym('x0', self.nx)
        # control input
        u = SX.sym('u', self.nu)

        # Append constraint to enforce initial state
        g0 = X[:, 0] - x_init
        g.append(g0)

        # Append collocation constraints
        for k in range(1, self.K + 1):
            gk = -self.dt * pendulum_model.systemdynamics(X[:, k], u)
            for j in range(self.K + 1):
                gk += A[j, k] * X[:, j]

            g.append(gk)

        # concatenate constraints to a vector
        g = vertcat(*g)

        # nlpsol Object
        nlp = {'x': X.reshape((-1, 1)), 'g': g, 'p': vertcat(x_init, u)}
        S = nlpsol('S', 'ipopt', nlp)

        return S

    def simulate_with_input_vector_and_OC(self, U_k, n_sim):
        S = self._simulateWithOC(pendulum_model.x, pendulum_model.u)

        x_0 = self.x0
        res_x_OC = [x_0]

        for i in range(n_sim):

            res = S(lbg=0, ubg=0, p=vertcat(x_0, U_k[i]))
            X_k = res['x'].full().reshape(self.K + 1, self.nx)
            x_next = X_k.T @ self.D
            res_x_OC.append(x_next)
            x_0 = x_next

        # Make an array from the list of arrays:
        res_x_OC = np.concatenate(res_x_OC, axis=1)

        return res_x_OC

    def simulate_with_input_vector_and_sundials(self, U_k, n_sim):

        x_0 = self.x0
        res_x_sundials = [x_0]

        for i in range(n_sim):
            res_integrator = self.ode_solver(x0=x_0, p=U_k[i])
            x_next = res_integrator['xf']
            res_x_sundials.append(x_next)
            x_0 = x_next

        # Make an array from the list of arrays:
        res_x_sundials = np.concatenate(res_x_sundials, axis=1)

        return res_x_sundials

    def _check_state_difference(self, res_x_mpc, tolerance):

        res_x_mpc = np.concatenate(res_x_mpc, axis=1)

        method = 'm1'

        if method == 'm1':  #only penalize velocities

            check_states = res_x_mpc[3:, -10:]
            norm = np.linalg.norm(check_states, axis=0)

            #print(sum(norm))

            if sum(norm) <= tolerance:
                return True

        if method == 'm2':

            xz_goal = np.zeros((6, 1))

            xz_goal[0] == self.xz_pos

            if self.xz_ori == 'down':
                xz_goal[1:3] = pi

            check_states = res_x_mpc[:, -10:] - xz_goal

            norm = np.linalg.norm(check_states, axis=0)

            print(sum(norm))

            if sum(norm) <= tolerance:
                return True

        return False

    def mpc_loop(self):

        # Initialize result lists for states and inputs
        res_x_mpc = [self.x0]
        res_u_mpc = []
        cost_res = []
        loop_pred = []

        #warmstart
        x_0 = self.x0
        initial_guess = self.opt_x(0)

        initial_guess['x'] = x_0

        stop_parameter = False

        # Set number of iterations

        for i in range(self.mpc_steps):
            # solve optimization problem

            mpc_res = self.mpc_solver(p=x_0,
                                      x0=initial_guess,
                                      lbg=self.lb_g,
                                      ubg=self.ub_g,
                                      lbx=self.lb_opt_x,
                                      ubx=self.ub_opt_x)

            # Extract the control input
            opt_x_k = self.opt_x(mpc_res['x'])
            u_k = opt_x_k['u', 0]

            initial_guess = opt_x_k

            x_loop_pred_now = np.array(horzcat(*opt_x_k['x', :,
                                                        0, :]))[:, 0:self.N]
            u_loop_pred_now = np.array([opt_x_k['u', :]])
            loop_pred_now = np.concatenate((u_loop_pred_now, x_loop_pred_now),
                                           axis=0)

            # simulate the system
            res_integrator = self.ode_solver(x0=x_0, p=u_k)
            x_next = res_integrator['xf']

            # simulation with oc (doesn't improve it)
            #S = self._simulateWithOC(pendulum_model.x, pendulum_model.u)
            #res_oc = S(lbg=0, ubg=0, p=vertcat(x_0, u_k))
            #X_k = res_oc['x'].full().reshape(self.K + 1, self.nx)
            #x_next = X_k.T @ self.D

            # Update the initial state
            x_0 = x_next

            # Store the results
            res_x_mpc.append(x_next)
            res_u_mpc.append(u_k)
            cost_res.append(self.stage_cost_fcn(x_next, u_k))
            loop_pred.append(loop_pred_now)

            if len(res_x_mpc) >= 10:

                stop_parameter = self._check_state_difference(res_x_mpc, 1.5)

            if stop_parameter == True:
                break

            print("____________________________________________________")
            print("iteration: ", i)
            print("____________________________________________________")

        # Make an array from the list of arrays
        res_x_mpc = np.concatenate(res_x_mpc, axis=1)
        res_u_mpc = np.concatenate(res_u_mpc, axis=1)
        cost_res = np.concatenate(cost_res, axis=1)

        return res_x_mpc, res_u_mpc, cost_res[0], loop_pred