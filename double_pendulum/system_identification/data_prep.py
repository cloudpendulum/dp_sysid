import numpy as np
import os
from scipy import signal

from double_pendulum.filter.offline.lowpass import lowpass_filter_offline
from double_pendulum.filter.offline.butterworth import butterworth_filter_offline
from double_pendulum.system_identification.plotting import plot_filtered_vs_raw_data

def smooth_data(t,
                shoulder_pos,
                shoulder_vel,
                shoulder_trq,
                elbow_pos,
                elbow_vel,
                elbow_trq,
                filt="butterworth"):

    if filt == "butterworth":
        Wn = 0.02
        filtered_shoulder_vel = butterworth_filter_offline(shoulder_vel, 3, Wn)
        filtered_elbow_vel = butterworth_filter_offline(elbow_vel, 3, Wn)
        filtered_shoulder_trq = butterworth_filter_offline(shoulder_trq, 3, Wn)
        filtered_elbow_trq = butterworth_filter_offline(elbow_trq, 3, Wn)

        filtered_shoulder_pos = butterworth_filter_offline(shoulder_pos, 3, Wn)
        filtered_elbow_pos = butterworth_filter_offline(elbow_pos, 3, Wn)

        # compute acceleration from positions and filter 2x
        vel1 = np.gradient(filtered_shoulder_pos, t)
        vel2 = np.gradient(filtered_elbow_pos, t)

        # vel1 = butterworth_filter_offline(vel1, 3, Wn)
        # vel2 = butterworth_filter_offline(vel2, 3, Wn)
        acc1=np.gradient(np.gradient(shoulder_pos,t),t)
        acc2=np.gradient(np.gradient(elbow_pos,t),t)

        filtered_shoulder_acc = np.gradient(filtered_shoulder_vel,t)
        filtered_elbow_acc = np.gradient(filtered_elbow_vel,t)
        # filtered_shoulder_acc = np.gradient(vel1,t)
        # filtered_elbow_acc = np.gradient(vel2,t)
        # filtered_shoulder_acc = acc1
        # filtered_elbow_acc = acc2

        # filtered_shoulder_acc = butterworth_filter_offline(filtered_shoulder_acc, 3, Wn)
        # filtered_elbow_acc = butterworth_filter_offline(filtered_elbow_acc, 3, Wn)
        # raw_shoulder_acc = np.gradient(np.gradient(shoulder_pos, t), t)
        # raw_elbow_acc = np.gradient(np.gradient(elbow_pos, t), t)
        print("filtered")
        save_dir="./results/"
        plot_filtered_vs_raw_data(
            t,
            shoulder_vel, filtered_shoulder_vel,
            elbow_vel, filtered_elbow_vel,
            # acc1, filtered_shoulder_acc,
            # acc2, filtered_elbow_acc,
            filtered_shoulder_acc, filtered_shoulder_acc,
            filtered_elbow_acc, filtered_elbow_acc,
            shoulder_trq, filtered_shoulder_trq,
            elbow_trq, filtered_elbow_trq,
            save_to=os.path.join(save_dir, "Filter.png")
        )
    # if filt == "butterworth":
    #     from scipy.signal import butter, filtfilt

    #     # Compute sampling frequency
    #     dt = t[1] - t[0]
    #     fs = 1.0 / dt
    #     cutoff = 5  # Hz (adjust as needed)
    #     order = 3

    #     b, a = butter(order, cutoff, fs=fs)

    #     # Filter pos, vel, trq
    #     filtered_shoulder_pos = filtfilt(b, a, shoulder_pos)
    #     filtered_elbow_pos = filtfilt(b, a, elbow_pos)
    #     filtered_shoulder_vel = filtfilt(b, a, shoulder_vel)
    #     filtered_elbow_vel = filtfilt(b, a, elbow_vel)
    #     filtered_shoulder_trq = filtfilt(b, a, shoulder_trq)
    #     filtered_elbow_trq = filtfilt(b, a, elbow_trq)

    #     # Compute acceleration from filtered velocity
    #     ddq1 = np.gradient(filtered_shoulder_vel, t)
    #     ddq2 = np.gradient(filtered_elbow_vel, t)

    #     # Filter acceleration (optional)
    #     ddq1_filt = filtfilt(b, a, ddq1)
    #     ddq2_filt = filtfilt(b, a, ddq2)

    #     filtered_shoulder_acc = ddq1_filt
    #     filtered_elbow_acc = ddq2_filt

    #     print("Filtered signals.")
    #     plot_filtered_vs_raw_data(
    #         t,
    #         shoulder_vel, filtered_shoulder_vel,
    #         elbow_vel, filtered_elbow_vel,
    #         filtered_shoulder_acc, filtered_shoulder_acc,
    #         filtered_elbow_acc, filtered_elbow_acc,
    #         shoulder_trq, filtered_shoulder_trq,
    #         elbow_trq, filtered_elbow_trq,
    #         save_to=os.path.join("./results/", "Filter.png")
    #     )


    elif filt == "lowpass":
        filtered_shoulder_vel = lowpass_filter_offline(shoulder_vel, 0.3)
        filtered_elbow_vel = lowpass_filter_offline(elbow_vel, 0.3)
        filtered_shoulder_trq = lowpass_filter_offline(shoulder_trq, 0.3)
        filtered_elbow_trq = lowpass_filter_offline(elbow_trq, 0.3)

        # compute acceleration from positions and filter 2x
        vel1 = np.gradient(shoulder_pos, t)
        vel2 = np.gradient(elbow_pos, t)

        vel1 = lowpass_filter_offline(vel1, 0.3)
        vel2 = lowpass_filter_offline(vel2, 0.3)

        filtered_shoulder_acc = np.gradient(vel1, t)
        filtered_elbow_acc = np.gradient(vel2, t)

        filtered_shoulder_acc = lowpass_filter_offline(filtered_shoulder_acc, 0.5)
        filtered_elbow_acc = lowpass_filter_offline(filtered_elbow_acc, 0.5)
    else:
        filtered_shoulder_vel = shoulder_vel
        filtered_elbow_vel = elbow_vel
        filtered_shoulder_trq = shoulder_trq
        filtered_elbow_trq = elbow_trq
        
        vel1 = np.gradient(shoulder_pos, t)
        vel2 = np.gradient(elbow_pos, t)

        filtered_shoulder_acc = np.gradient(vel1, t)
        filtered_elbow_acc = np.gradient(vel2, t)


    return (t,
            shoulder_pos,
            elbow_pos,
            filtered_shoulder_vel,
            filtered_elbow_vel,
            filtered_shoulder_acc,
            filtered_elbow_acc,
            filtered_shoulder_trq,
            filtered_elbow_trq)
