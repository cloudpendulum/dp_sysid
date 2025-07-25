import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_torques(time,
                 shoulder_meas_tau,
                 elbow_meas_tau,
                 shoulder_fit_tau,
                 elbow_fit_tau,
                 save_to=None,
                 show=True):

    fig, ax = plt.subplots(2, 2,
                           figsize=(18, 6),
                           sharex="all",
                           sharey="row")

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
    mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ax[0, 0].plot(time, shoulder_meas_tau, label="u1 (measured)", color="blue")
    ax[0, 0].plot(time, shoulder_fit_tau, ls="--", label="u1 (fit)", color="lightblue")
    ax[0, 0].set_ylabel("torque [Nm]", fontsize=MEDIUM_SIZE)
    ax[0, 0].legend(loc="best", fontsize=MEDIUM_SIZE)
    ax[0, 0].set_title("shoulder joint", fontsize=MEDIUM_SIZE)

    ax[0, 1].plot(time, elbow_meas_tau, label="u2 (measured)", color="red")
    ax[0, 1].plot(time, elbow_fit_tau, ls="--", label="u2 (fit)", color="orange")
    ax[0, 1].legend(loc="best", fontsize=MEDIUM_SIZE)
    ax[0, 1].set_title("elbow joint", fontsize=MEDIUM_SIZE)

    ax[1, 0].plot(time, shoulder_fit_tau - shoulder_meas_tau, color="blue")
    ax[1, 0].set_xlabel("time [s]", fontsize=MEDIUM_SIZE)
    ax[1, 0].set_ylabel("torque diff", fontsize=MEDIUM_SIZE)

    ax[1, 1].plot(time, elbow_fit_tau - elbow_meas_tau, color="red")
    #ax[1, 1].set_ylim(-10, 10)
    ax[1, 1].set_xlabel("time [s]", fontsize=MEDIUM_SIZE)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.05,
                        hspace=0.05)

    if not (save_to is None):
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.show()
        plt.close()

def plot_filtered_vs_raw_data(t,
                               raw_shoulder_vel, filtered_shoulder_vel,
                               raw_elbow_vel, filtered_elbow_vel,
                               raw_shoulder_acc, filtered_shoulder_acc,
                               raw_elbow_acc, filtered_elbow_acc,
                               raw_shoulder_trq, filtered_shoulder_trq,
                               raw_elbow_trq, filtered_elbow_trq,
                               save_to=None):
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Filtered vs Noisy Data Comparison")

    axs[0, 0].plot(t, raw_shoulder_vel, label="Raw", alpha=0.5)
    axs[0, 0].plot(t, filtered_shoulder_vel, label="Filtered")
    axs[0, 0].set_title("Shoulder Velocity")
    axs[0, 0].legend()

    axs[0, 1].plot(t, raw_elbow_vel, label="Raw", alpha=0.5)
    axs[0, 1].plot(t, filtered_elbow_vel, label="Filtered")
    axs[0, 1].set_title("Elbow Velocity")
    axs[0, 1].legend()

    axs[1, 0].plot(t, raw_shoulder_acc, label="Raw", alpha=0.5)
    axs[1, 0].plot(t, filtered_shoulder_acc, label="Filtered")
    axs[1, 0].set_title("Shoulder Acceleration")
    axs[1, 0].legend()

    axs[1, 1].plot(t, raw_elbow_acc, label="Raw", alpha=0.5)
    axs[1, 1].plot(t, filtered_elbow_acc, label="Filtered")
    axs[1, 1].set_title("Elbow Acceleration")
    axs[1, 1].legend()

    axs[2, 0].plot(t, raw_shoulder_trq, label="Raw", alpha=0.5)
    axs[2, 0].plot(t, filtered_shoulder_trq, label="Filtered")
    axs[2, 0].set_title("Shoulder Torque")
    axs[2, 0].legend()

    axs[2, 1].plot(t, raw_elbow_trq, label="Raw", alpha=0.5)
    axs[2, 1].plot(t, filtered_elbow_trq, label="Filtered")
    axs[2, 1].set_title("Elbow Torque")
    axs[2, 1].legend()

    for ax in axs.flat:
        ax.grid(True)

    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 1].set_xlabel("Time [s]")

    plt.tight_layout()
    if not (save_to is None):
        plt.savefig(save_to, bbox_inches="tight")
    plt.show()

