import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib import transforms
import matplotlib.animation as animation
import numpy as np

l1 = 0.5  # m, length of the first rod
l2 = 0.5  # m, length of the second rod
animation_scale = 3
dt = 0.04

mario_mode = True
coin_positions = [-6.5, 0, 7]
coin_remove_time = [150, 300, 420]


def drawCoin(ax, pos_x):
    coin = plt.Circle((pos_x, 0.5 - 3.2),
                      radius=0.4,
                      fill=True,
                      color='yellow',
                      zorder=2)
    coin_fill1 = plt.Circle((pos_x, 0.5 - 3.2),
                            radius=0.3,
                            fill=False,
                            color='black',
                            zorder=3)
    coin_fill2 = plt.Rectangle((pos_x - 0.05, 0.3 - 3.2),
                               0.1,
                               0.4,
                               fill=False,
                               color='black',
                               zorder=3)

    ax.add_patch(coin)
    ax.add_patch(coin_fill1)
    ax.add_patch(coin_fill2)

    return [coin, coin_fill1, coin_fill2]


def removeCoin(coin):
    for elem in coin:
        elem.set_visible(False)


def animate(i):

    # animation plot
    thisx = [x0[i], x1[i], x2[i]]
    thisy = [y0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    circle1.set_center([x0[i] - 0.3, -0.162 + y0])
    circle2.set_center([x0[i] + 0.3, -0.162 + y0])
    rect.set_x(x0[i] - 0.45)

    # u plot
    u_graph.set_data(this_discretization[0:i + 1],
                     this_control_inputs[0:i + 1])
    pred_u_graph.set_data(
        this_discretization[i + 1:i + 1 + this_predictions[i][0].size],
        this_predictions[i][0])

    # pos plot
    pos_graph1.set_data(this_discretization[0:i + 1], this_states[:,
                                                                  0][0:i + 1])
    pred_pos_graph1.set_data(
        this_discretization[i:i + this_predictions[i][1].size],
        this_predictions[i][1])

    pos_graph2.set_data(this_discretization[0:i + 1], this_states[:,
                                                                  1][0:i + 1])
    pred_pos_graph2.set_data(
        this_discretization[i:i + this_predictions[i][2].size],
        this_predictions[i][2])

    pos_graph3.set_data(this_discretization[0:i + 1], this_states[:,
                                                                  2][0:i + 1])
    pred_pos_graph3.set_data(
        this_discretization[i:i + this_predictions[i][3].size],
        this_predictions[i][3])

    # vel plot
    vel_graph1.set_data(this_discretization[0:i + 1], this_states[:,
                                                                  3][0:i + 1])
    pred_vel_graph1.set_data(
        this_discretization[i:i + this_predictions[i][4].size],
        this_predictions[i][4])

    vel_graph2.set_data(this_discretization[0:i + 1], this_states[:,
                                                                  4][0:i + 1])
    pred_vel_graph2.set_data(
        this_discretization[i:i + this_predictions[i][5].size],
        this_predictions[i][5])

    vel_graph3.set_data(this_discretization[0:i + 1], this_states[:,
                                                                  5][0:i + 1])
    pred_vel_graph3.set_data(
        this_discretization[i:i + this_predictions[i][6].size],
        this_predictions[i][6])

    if mario_mode:
        for j in range(len(coins)):
            if coin_remove_time[j] == i:
                removeCoin(coins[j])


def animate_state_sequence(states, control_inputs, predictions, constraints):
    """

    Args:
        states: numpy array of shape (number_of_timesteps, 6) 
                with states in first array being x, theta1, theta2, x_dot, theta1_dot, theta2_dot
        control_inputs: numpy array of shape (number_of_timesteps,)
        predictions: numpy array of shape (number_of_timesteps, 7, prediction_horizon) 
                with second array being u, x, theta1, theta2, x_dot, theta1_dot, theta2_dot
    """

    # make variables global for animate function
    global this_states, this_control_inputs, this_discretization, this_predictions
    this_states = states
    this_control_inputs = control_inputs
    this_predictions = predictions
    this_discretization = np.linspace(
        0, dt * (this_control_inputs.size + this_predictions[0][0].size),
        this_control_inputs.size + this_predictions[0][0].size + 1)

    # create plots
    fig = plt.figure(figsize=(16, 8))
    ani_ax = fig.add_axes([0., 0., .6, 1])
    vel_ax = fig.add_axes([.63, 0 + 0.03, .35, (1 / 3) - 0.06])
    pos_ax = fig.add_axes([.63, (1 / 3) + 0.03, .35, (1 / 3) - 0.06])
    u_ax = fig.add_axes([.63, (2 / 3) + 0.03, .35, (1 / 3) - 0.06])

    # animation plot
    global x0, y0, x1, y1, x2, y2, line, rect, circle1, circle2, coins
    ani_ax.set_xlim([animation_scale * (-3), animation_scale * 3])
    ani_ax.set_ylim([animation_scale * (-2.5), animation_scale * 2.5])
    ani_ax.set_axis_off()

    y0 = 0
    if mario_mode:
        coins = []
        for pos in coin_positions:
            coin = drawCoin(ani_ax, pos)
            coins.append(coin)
        y0 = -4
        with cbook.get_sample_data(
                '/home/jan/Desktop/MPC/NMPC_double_inverted_pendulum/pictures/supermario.png'
        ) as image_file:
            #/home/jan/Desktop/MPC/NMPC_double_inverted_pendulum/pictures/supermario.png
            #/Users/WelfRehberg/Documents/GitHub/NMPC_double_inverted_pendulum/pictures/supermario.png
            image = plt.imread(image_file)
        ani_ax.imshow(image, extent=[-10, 10, -7.15, 6], alpha=0.66)
        ground = ani_ax.plot(
            [-10, 10],
            [-0.162 - 0.07 + y0, -0.162 - 0.07 + y0],
            color='coral',
            lw=1,
        )
    else:
        ani_ax.grid(zorder=1)
        ground = ani_ax.plot([-10, 10], [-0.162 - 0.07, -0.162 - 0.07],
                             color='g')

    line, = ani_ax.plot([], [], 'o-', lw=3, color='blue', zorder=10)
    circle1 = plt.Circle((-0.3, -0.162),
                         radius=0.07,
                         fill=True,
                         color='k',
                         zorder=2)
    circle2 = plt.Circle((0.3, -0.162),
                         radius=0.07,
                         fill=True,
                         color='k',
                         zorder=3)
    rect = plt.Rectangle((-0.45, -0.15 + y0),
                         0.9,
                         0.3,
                         fill=True,
                         color='dimgrey',
                         zorder=4)
    ani_ax.add_patch(circle1)
    ani_ax.add_patch(circle2)
    #ani_ax.add_patch(sun)
    ani_ax.add_patch(rect)
    x0 = states[:, 0]
    x1 = l1 * np.sin(states[:, 1]) + x0
    y1 = l1 * np.cos(states[:, 1]) + y0
    x2 = l2 * np.sin(states[:, 2]) + x1
    y2 = l2 * np.cos(states[:, 2]) + y1
    #constraints
    color = ''
    if mario_mode:
        color = 'black'
    else:
        color = 'red'
    for constraint in constraints:
        constraint_patch = plt.Circle([
            constraint['center'][0],
            constraint['center'][1] + y0,
        ],
                                      constraint['radius'] - 0.15,
                                      fill=False,
                                      hatch='/',
                                      color=color,
                                      linewidth=5)
        ani_ax.add_patch(constraint_patch)

    global u_graph, pos_graph1, pos_graph2, pos_graph3, vel_graph1, vel_graph2, vel_graph3
    # for predictions
    global pred_u_graph, pred_pos_graph1, pred_pos_graph2, pred_pos_graph3, pred_vel_graph1, pred_vel_graph2, pred_vel_graph3

    # u plot
    u_ax.get_xaxis().set_visible(False)
    u_ax.set(xlim=(0, dt *
                   (this_control_inputs.size + this_predictions[0][0].size)),
             ylim=(-5, 5))
    u_ax.set_title('control inputs')

    u_graph, = u_ax.step([], [], color='b')
    pred_u_graph, = u_ax.plot([], [], color='b', linestyle=':')

    # pose plot
    pos_ax.get_xaxis().set_visible(False)
    pos_ax.set(xlim=(0, dt *
                     (this_control_inputs.size + this_predictions[0][0].size)),
               ylim=(-np.pi * 4.3, np.pi * 4.3))
    pos_ax.set_title('pose states')
    pos_graph1, = pos_ax.plot([], [], color='g', label='x')
    pred_pos_graph1, = pos_ax.plot([], [], color='g', linestyle=':')
    pos_graph2, = pos_ax.plot([], [], color='m', label=r'$\theta_1$')
    pred_pos_graph2, = pos_ax.plot([], [], color='m', linestyle=':')
    pos_graph3, = pos_ax.plot([], [], color='r', label=r'$\theta_2$')
    pred_pos_graph3, = pos_ax.plot([], [], color='r', linestyle=':')
    pos_ax.legend()

    # velocities plot
    vel_ax.set(xlim=(0, dt *
                     (this_control_inputs.size + this_predictions[0][0].size)),
               ylim=(-23, 23))
    vel_ax.set_title('velocity states')
    vel_graph1, = vel_ax.plot([], [], color='g', label='x')
    pred_vel_graph1, = vel_ax.plot([], [], color='g', linestyle=':')
    vel_graph2, = vel_ax.plot([], [], color='m', label=r'$\theta_1$')
    pred_vel_graph2, = vel_ax.plot([], [], color='m', linestyle=':')
    vel_graph3, = vel_ax.plot([], [], color='r', label=r'$\theta_2$')
    pred_vel_graph3, = vel_ax.plot([], [], color='r', linestyle=':')
    vel_ax.legend()

    # animate!
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  interval=10,
                                  repeat=False,
                                  save_count=427)  #, init_func=init)
    plt.tight_layout()
    plt.show()

    Writer = animation.writers['ffmpeg']
    #writer = animation.FFMpegFileWriter(metadata=dict(artist='Me'), fps=1, bitrate=50000)
    writer = Writer(metadata=dict(artist='Me'),
                    fps=15,
                    bitrate=5000,
                    codec="libx264",
                    extra_args=['-pix_fmt', 'yuv420p'])
    ani.save('DIP_complex_path2.mp4', writer=writer)


def visualize_state_sequence(states, control_inputs):
    """

    Args:
        states: numpy array of shape (6, number_of_timesteps)
                with states in first array being x, theta1, theta2, x_dot, theta1_dot, theta2_dot
        control_inputs: numpy array of shape (number_of_timesteps,)
    """

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    # plot inputs
    ax[0].plot(np.array(states))
    ax[1].plot(np.array(control_inputs))

    # Set labels
    ax[0].set_ylabel('states')
    ax[0].set_xlabel('time')
    ax[0].legend(
        ('x', 'theta1', 'theta2', 'x_dot', 'theta1_dot', 'theta2_dot'))
    ax[1].set_ylabel('inputs')
    ax[1].set_xlabel('time')

    plt.show()


def visualize_state_sequence_costf(states, control_inputs, cost):
    """

    Args:
        states: numpy array of shape (6, number_of_timesteps)
                with states in first array being x, theta1, theta2, x_dot, theta1_dot, theta2_dot
        control_inputs: numpy array of shape (number_of_timesteps,)
    """

    fig, ax = plt.subplots(3, 1, figsize=(10, 6))

    # plot inputs
    ax[0].plot(np.array(states))
    ax[1].plot(np.array(control_inputs))
    ax[2].plot(np.array(cost))

    # Set labels
    ax[0].set_ylabel('states')
    ax[0].legend(('x', 'theta1', 'theta2', 'x_p', 'theta1_p', 'theta2_p'))
    ax[1].set_ylabel('inputs')
    ax[1].set_xlabel('time')
    ax[2].set_ylabel('cost')
    ax[2].set_xlabel('time')

    plt.show()
