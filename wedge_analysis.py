import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from IPython.core.debugger import Tracer
import read_videos as rv
import pickle
import copy

debug_here = Tracer()

plt.close("all")

# define constants
g = 9.81 * 100  # cm/s2
beta = 0.8e-3  # haline contraction co-eff (1/psu)


# pset = 1


# function to compute froude number given q where q is volume flux (m3/s) divided channel width (m)
def froude_q(q, g_p, h):
    return q / np.sqrt(g_p * h ** 3)


# define model for interface thickness:
def interface_shape2(t, y, q, c_i, c_d, gp, h_full, w_full):
    # t=x, y=hi=h2, assuming flat bottom

    h2 = y[0]
    h1 = h_full - h2
    Fr = froude_q(q, gp, h1)
    # dh = (c_i * (Fr**2 / (1 - y_p)) + c_d * Fr**2) / (y_p**3 - Fr**2)
    dh = (Fr ** 2 / (1 - Fr ** 2)) * (-c_i * h_full / h2 - c_d * (1 + 2 * h1 / w_full))

    return [dh]


def integrate_wedge(pset, ci, cd, ds=33.5):
    # plt.close("all")
    # Define flow parameters following example in Geyer and Ralston
    h0 = 2.5  # tube height (cm)
    w0 = 2.5  # tube width (cm)
    # s2 = 33  # wedge salinity (psu)
    # s1 = 0  # freshwater salinity
    ds = ds

    if pset == 1:
        C_i = 1e-1
        C_d = 1e-1

    elif pset == 1.1:
        C_i = 1e-4
        C_d = 1e-1

    elif pset == 1.2:
        C_i = 1e-1
        C_d = 1e-4

    elif pset == 1.3:
        C_i = 1e-1
        C_d = 0

    elif pset == 1.4:
        C_i = 0
        C_d = 1e-1

    elif pset == 1.5:
        C_i = 0
        C_d = 5e-1

    elif pset == 1.6:
        C_i = 5e-1
        C_d = 0

    elif pset == 0:
        C_i = ci
        C_d = cd

    Q1_arr = np.arange(5, 20, 2.5)  # cm3/s
    q1_arr = Q1_arr / w0

    # get interface for different outflow speeds
    hi_ls = []
    F1_ls = []
    x_ls = []
    for q1 in q1_arr:

        # define initial h1 so that Froude number = 0.9999 - just below critical
        F1_0 = 0.9999
        g_p = ds * beta * g
        h1_0 = (q1 ** 2 / (F1_0 ** 2 * g_p)) ** (1. / 3)
        h2_0 = h0 - h1_0

        # define initial conditions
        hi_init = np.array([h2_0])
        x_end = 1000  # cm

        # load and intialize interface model
        dx = 0.1  # cm
        ode = spi.ode(interface_shape2)  #
        ode.set_f_params(q1, C_i, C_d, g_p, h0, w0)

        ode.set_integrator('lsoda')
        ode.set_initial_value(hi_init, t=0)

        # debug_here()

        ts = []
        ys = []
        # intergrate interface equation
        while ode.successful() and ode.t < x_end and ode.y[0] > 0.1:
            # ode.t is the independent variable (x)
            # ode.y is the solution at x (h_i)
            ode.integrate(ode.t + dx)
            ts.append(ode.t)
            ys.append(ode.y[0])

        x = np.array(ts)
        hi = np.array(ys)
        F1 = froude_q(q1, g_p, (h0 - hi))

        x_ls.append(x)
        hi_ls.append(hi)
        F1_ls.append(F1)

    fig1 = plt.figure(figsize=(11, 6))
    wedge_L_max = []
    for i in range(len(F1_ls)):
        plt.subplot(211)
        plt.plot(x_ls[i], np.array(hi_ls[i]), label='Q1= %.2f m3/s' % Q1_arr[i])
        plt.xlabel("Along tube distance (cm)")
        plt.ylabel("Interface height (cm)")
        plt.title("Wedge shape")
        plt.legend(loc=0, fontsize=8, ncol=2)
        plt.xlim(0, 100)
        plt.grid(True)

        plt.subplot(212)
        plt.plot(x_ls[i], np.array(F1_ls[i]), label='Q1= %.2f m3/s' % Q1_arr[i])
        plt.xlabel("Along tube distance (cm)")
        plt.ylabel("F1")
        plt.legend(loc=0, fontsize=8, ncol=2)
        plt.grid(True)
        plt.title("Local Froude Number")
        plt.xlim(0, 100)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8, ncol=2)

        plt.suptitle("w0=%scm, h0=%scm, $C_i=%s$ and $C_d=%s$ " % (w0, h0, C_i, C_d), fontsize=14)
        plt.subplots_adjust(hspace=0.4)
        plt.savefig("analysis/integrated_wedge_stats_pset%s.pdf" % pset)

        # debug_here()
        wedge_L_max.append(np.max(x_ls[i]))

    wedge_L_max = np.array(wedge_L_max)

    # plot max wedge length versus pump speed
    fig2 = plt.figure()
    plt.plot(Q1_arr, wedge_L_max, lw=2)
    plt.grid(True)
    plt.ylabel("Wedge length (cm)")
    plt.xlabel("Volume Flux (cm3/s)")
    plt.title("w0=%scm, h0=%scm, $C_i=%s$ and $C_d=%s$ " % (w0, h0, C_i, C_d), fontsize=14)
    plt.savefig("analysis/wedge_length_vs_Q_pset%s.pdf" % pset)

    plt.close(fig1)
    plt.close(fig2)

    return np.array(Q1_arr), wedge_L_max


def plot_wedge_length_vs_qflux(run_name):

    # load data (TODO: use all available data)
    lab_run = pickle.load(open("analysis/%s_wedge_stats.p" %run_name, "rb"))
    img_params = rv.get_image_settings(run_name)

    # define parameter space
    w0 = 2.5
    h0 = 2.5
    # c_i = np.atleast_2d(np.array([1e-2, 2e-2, 4e-2, 1e-1]))
    # c_d = np.atleast_2d(np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))
    c_d = np.atleast_2d(np.array([6e-2, 1e-1, 1.4e-1, 1.8e-1, 2e-1]))
    c_i = 0*np.ones(c_d.shape)

    # c_i_2d = np.tile(c_i, (len(c_i.flatten()), 1))
    # c_d_2d = np.tile(c_d, (len(c_i.flatten()), 1)).T
    # c_i_d = np.array(list(zip(c_i_2d.flatten(), c_d_2d.flatten())))

    c_i_d = np.hstack((c_i.T, c_d.T))
    sort_i = np.argsort(np.sum(c_i_d, axis=1))
    c_i_d_sorted = c_i_d[sort_i, :]

    debug_here()

    # plot max wedge length versus pump speed
    plt.figure(figsize=(11, 8))

    # define color cycle
    cmap1 = copy.copy(plt.get_cmap('rainbow'))
    num_colors1 = len(c_i_d_sorted)
    colors1 = cmap1(np.linspace(0, 1, num_colors1))


    for i, (ci, cd) in enumerate(c_i_d_sorted):

        print("%s: %s %s" %(i, ci, cd))

        Q1_arr, wedge_L_max = integrate_wedge(0, ci, cd)

        plt.plot(Q1_arr, wedge_L_max, lw=2, color=colors1[i], label="$C_i=%s$, $C_d=%s$" % (ci, cd))

    # add data
    # debug_here()
    plt.plot(img_params['pump_flux'], lab_run["w_len_eq"], 'o', color='0.5', markersize=8, linewidth=2,
             label=lab_run['run_name'], alpha=0.5)

    plt.grid(True)
    plt.ylabel("Wedge length (cm)")
    plt.xlabel("Volume Flux (cm3/s)")
    # plt.legend(loc=0, ncol=2, fontsize=10)
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8, ncol=1)

    plt.ylim(0, 100)

    plt.title("w0=%scm, h0=%scm " % (w0, h0), fontsize=14)

    plt.show()
    plt.savefig("analysis/wedge_length_vs_Q.pdf", bbox_inches='tight')







    # function to compute froude number given u
    # def Froude_u(u, g_p, h):
    #
    #     return u / np.sqrt(g_p * h)

    # define model for interface thickness (assuming estuarine parameters):
    # def interface_shape1(t, y, u1, C_i, ds, h0):
    #     # t=x, y=hi=h2, assuming flat bottom
    #     h1 = h0 - y[0]
    #     F1 = Froude_q(u1, g, ds, beta, h1)
    #     dh = -C_i * (F1 ** 2 / (1 - F1 ** 2)) * (h0 / y[0])
    #
    #     return [dh]
