import visualization
import controller
import numpy as np


def main():

    # # # # # # # # # # # #
    #    Hyperparameter   #
    # # # # # # # # # # # #
    # Costfunction
    Q = np.array([0, 1, 0])  # Stage costs
    R = 0.001  # Input costs
    P = np.array([1, 1, 1])  # Terminal costs
    # Simulation related
    dt = 0.04
    N = 15
    mpc_steps = int(8 / dt)  # x is time in seconds
    K = 2  # collocation variable

    # # # # # # # # # # # #
    #        Poses        #
    # # # # # # # # # # # #
    # start pose
    x0 = np.zeros((6, 1))
    x0[0] = 0
    x0[1] = -np.pi + 0.01  # funktioniert aber mit x0[1]=np.pi; x0[2] = np.pi
    x0[2] = -np.pi + 0.01
    # goal pose
    xz_pos = 2  # translation in x
    xz_ori = 'up_E'  # up or down

    # # # # # # # # # # # #
    #  Circle Constraints #
    # # # # # # # # # # # #
    circles = []

    # # # # # # # # # # # #
    #   Control tasks     #
    # # # # # # # # # # # #
    mode = 's2'

    # s1 and s2 were used in the evaluation part of the paper

    print(
        'You chose to execute mode {}. Make sure to change the picture-path in visualization.py to be able to use mario mode!'
        .format(mode))
    input("Press something to continue...")

    if mode == 's1':

        Q = np.array([0, 1, 0.1])
        Q[2] = 1
        N = 50
        mpc_steps = int(8 / dt)
        x0[1] = np.pi
        x0[2] = np.pi
        P = np.array([5, 1, 1, 1])

        #obstacles
        circles.append({"center": [-3.0, 1.0], "radius": 0.5})
        circles.append({"center": [-3.0, -1.0], "radius": 0.5})

        # Controller 1
        x0[0] = -6.0
        xz_pos = -6.0
        #x0[0] = 0
        #xz_pos = 0.5

        mpc_controller1 = controller.mpc_controller(Q, R, P, dt, N, mpc_steps,
                                                    K, x0, xz_pos, xz_ori,
                                                    circles)

        X1, U1, cost1, pred1 = mpc_controller1.mpc_loop()

        # Controller 2
        #xz_pos = 0
        #huepf
        xz_pos = -2
        x0 = np.array([X1[:, -1]]).T  # take last state from X1 as start state
        mpc_controller2 = controller.mpc_controller(Q, R, P, dt, N, mpc_steps,
                                                    K, x0, xz_pos, xz_ori,
                                                    circles)

        X2, U2, cost2, pred2 = mpc_controller2.mpc_loop()

        # concatenate data
        X = np.concatenate((X1, X2), axis=1)
        U = np.concatenate((U1, U2), axis=1)
        cost = np.concatenate((cost1, cost2), axis=0)
        pred = np.concatenate((pred1, pred2), axis=0)

        # visualization
        visualization.visualize_state_sequence_costf(X.T, U.T, cost)

        # animation
        U_correct_form = np.array(U)[0]
        X_correct_form = np.array(X).T
        visualization.animate_state_sequence(X_correct_form, U_correct_form,
                                             pred, circles)

    if mode == 's2':
        N = 30
        mpc_steps = int(8 / dt)
        x0[1] = np.pi
        x0[2] = np.pi

        #obstacles
        circles.append({"center": [-3.5, 1.3], "radius": 0.5})
        circles.append({"center": [-3.5, -1.3], "radius": 0.5})

        circles.append({"center": [-2.5, -1.3], "radius": 0.5})

        circles.append({"center": [3, 1.65], "radius": 0.8})
        circles.append({"center": [5, -1.3], "radius": 0.5})

        # Controller 1
        x0[0] = -6.5
        xz_pos = -6.5
        Q = np.array([0, 1, 0])  # Stage costs
        R = 0.001  # Input costs
        P = np.array([1, 1, 1, 1])  # Terminal costs

        mpc_controller1 = controller.mpc_controller(Q, R, P, dt, N, mpc_steps,
                                                    K, x0, xz_pos, xz_ori,
                                                    circles)

        X1, U1, cost1, pred1 = mpc_controller1.mpc_loop()

        # Controller 2.
        xz_pos = 0
        Q = np.array([0, 0.5, 0.8])  # Stage costs
        R = 0.001  # Input costs
        P = np.array([1, 1, 1, 1])  # Terminal costs

        x0 = np.array([X1[:, -1]]).T  # take last state from X1 as start state
        mpc_controller2 = controller.mpc_controller(Q, R, P, dt, N, mpc_steps,
                                                    K, x0, xz_pos, xz_ori,
                                                    circles)

        X2, U2, cost2, pred2 = mpc_controller2.mpc_loop()

        # Controller 3
        xz_pos = 7
        Q = np.array([0, 1, 0.8])  # Stage costs
        R = 0.001  # Input costs
        P = np.array([1, 1, 1, 1])  # Terminal costs

        x0 = np.array([X2[:, -1]]).T  # take last state from X2 as start state
        mpc_controller3 = controller.mpc_controller(Q, R, P, dt, N, mpc_steps,
                                                    K, x0, xz_pos, xz_ori,
                                                    circles)

        X3, U3, cost3, pred3 = mpc_controller3.mpc_loop()

        # concatenate data
        X = np.concatenate((X1, X2, X3), axis=1)
        U = np.concatenate((U1, U2, U3), axis=1)
        cost = np.concatenate((cost1, cost2, cost3), axis=0)
        pred = np.concatenate((pred1, pred2, pred3), axis=0)

        # visualization
        visualization.visualize_state_sequence_costf(X.T, U.T, cost)

        # animation
        U_correct_form = np.array(U)[0]
        X_correct_form = np.array(X).T
        visualization.animate_state_sequence(X_correct_form, U_correct_form,
                                             pred, circles)

    # these modes were not used in the evaluation part of the paper (experimental stuff)

    if mode == 'm1':  # Upswing

        mpc_controller = controller.mpc_controller(Q, R, P, dt, N, mpc_steps,
                                                   K, x0, xz_pos, xz_ori,
                                                   circles)

        #X, U, cost, pred = mpc_controller.mpc_firstprediction()
        X, U, cost, pred = mpc_controller.mpc_loop()

        #    Visualization
        visualization.visualize_state_sequence_costf(X.T, U.T, cost)

        # animation
        U_correct_form = np.array(U)[0]
        X_correct_form = np.array(X).T
        visualization.animate_state_sequence(X_correct_form, U_correct_form,
                                             pred, circles)

    if mode == 'm2':
        Q[2] = 1
        dt = 0.04
        mpc_steps = int(6 / dt)

        x0 = np.zeros((6, 1))
        x0[0] = -2.5
        xz_pos = 3.5
        N = 40
        circles.append({"center": [0.5, 1.2], "radius": 0.5})
        circles.append({"center": [0.5, -1.2], "radius": 0.5})
        mpc_controller = controller.mpc_controller(Q, R, P, dt, N, mpc_steps,
                                                   K, x0, xz_pos, xz_ori,
                                                   circles)

        #X, U, cost, pred = mpc_controller.mpc_firstprediction()
        X, U, cost, pred = mpc_controller.mpc_loop()

        #    Visualization
        visualization.visualize_state_sequence_costf(X.T, U.T, cost)

        # animation
        U_correct_form = np.array(U)[0]
        X_correct_form = np.array(X).T
        visualization.animate_state_sequence(X_correct_form, U_correct_form,
                                             pred, circles)

    if mode == 'm3':
        Q[2] = 1
        dt = 0.04
        mpc_steps = int(6 / dt)

        x0 = np.zeros((6, 1))
        x0[0] = -4.5
        xz_pos = 4.5
        N = 40
        circles.append({"center": [-2.5, 1.2], "radius": 0.5})
        circles.append({"center": [-0.5, 1.2], "radius": 0.5})
        circles.append({"center": [2, 1.2], "radius": 0.5})
        mpc_controller = controller.mpc_controller(Q, R, P, dt, N, mpc_steps,
                                                   K, x0, xz_pos, xz_ori,
                                                   circles)

        #X, U, cost, pred = mpc_controller.mpc_firstprediction()
        X, U, cost, pred = mpc_controller.mpc_loop()

        #    Visualization
        visualization.visualize_state_sequence_costf(X.T, U.T, cost)

        # animation
        U_correct_form = np.array(U)[0]
        X_correct_form = np.array(X).T
        visualization.animate_state_sequence(X_correct_form, U_correct_form,
                                             pred, circles)

    if mode == 'm4':
        Q[2] = 1
        dt = 0.04
        mpc_steps = int(6 / dt)

        x0 = np.zeros((6, 1))
        x0[0] = -4.5
        xz_pos = 4.5
        N = 50
        circles.append({"center": [-2.5, 1.2], "radius": 0.5})
        circles.append({"center": [-0.5, -1], "radius": 0.5})
        circles.append({"center": [2, 1.2], "radius": 0.5})
        mpc_controller = controller.mpc_controller(Q, R, P, dt, N, mpc_steps,
                                                   K, x0, xz_pos, xz_ori,
                                                   circles)

        #X, U, cost, pred = mpc_controller.mpc_firstprediction()
        X, U, cost, pred = mpc_controller.mpc_loop()

        #    Visualization
        visualization.visualize_state_sequence_costf(X.T, U.T, cost)

        # animation
        U_correct_form = np.array(U)[0]
        X_correct_form = np.array(X).T
        visualization.animate_state_sequence(X_correct_form, U_correct_form,
                                             pred, circles)


if __name__ == "__main__":
    main()
