import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation
import time

from double_pendulum.simulation.visualization import get_arrow, set_arrow_properties


class Simulator:
    def __init__(self, plant):
        self.plant = plant

        self.x = np.zeros(2 * self.plant.dof)  # position, velocity
        self.t = 0.0  # time

        self.desired_state = False
        self.desired_traj = False
        self.T_des = None
        self.X_des = None
        self.x_des = None

    def set_state(self, time, x):
        self.x = x
        self.t = time

    def get_state(self):
        return self.t, self.x

    def reset_data_recorder(self):
        self.t_values = []
        self.x_values = []
        self.tau_values = []

    def reset(self):
        self.reset_data_recorder()
        self.t = 0.0
        self.desired_state = False
        self.desired_traj = False
        self.T_des = None
        self.X_des = None
        self.x_des = None

    def record_data(self, time, x, tau):
        self.t_values.append(time)
        self.x_values.append(x)
        self.tau_values.append(tau)

    def euler_integrator(self, t, y, dt, tau):
        return self.plant.rhs(t, y, tau)

    def runge_integrator(self, t, y, dt, tau):
        k1 = self.plant.rhs(t, y, tau)
        k2 = self.plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k1, tau)
        k3 = self.plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k2, tau)
        k4 = self.plant.rhs(t + dt, y + dt * k3, tau)
        return (k1 + 2 * (k2 + k3) + k4) / 6.0

    def step(self, tau, dt, integrator="runge_kutta"):
        tau = np.clip(
            tau,
            -np.asarray(self.plant.torque_limit),
            np.asarray(self.plant.torque_limit),
        )

        self.record_data(self.t, self.x.copy(), tau)

        if integrator == "runge_kutta":
            self.x += dt * self.runge_integrator(self.t, self.x, dt, tau)
        elif integrator == "euler":
            self.x += dt * self.euler_integrator(self.t, self.x, dt, tau)
        else:
            raise NotImplementedError(
                f"Sorry, the integrator {integrator} is not implemented."
            )
        self.t += dt
        # self.record_data(self.t, self.x.copy(), tau)

    def simulate(self, t0, x0, tf, dt, controller=None, integrator="runge_kutta"):
        self.set_state(t0, x0)
        self.reset_data_recorder()

        while self.t <= tf:
            if controller is not None:
                tau = controller.get_control_output(x=self.x, t=self.t)
            else:
                tau = np.zeros(self.plant.n_actuators)
            self.step(tau, dt, integrator=integrator)

        return self.t_values, self.x_values, self.tau_values

    def _animation_init(self):
        """
        init of the animation plot
        """
        self.animation_ax.set_xlim(
            self.plant.workspace_range[0][0], self.plant.workspace_range[0][1]
        )
        self.animation_ax.set_ylim(
            self.plant.workspace_range[1][0], self.plant.workspace_range[1][1]
        )
        self.animation_ax.get_xaxis().set_visible(False)
        self.animation_ax.get_yaxis().set_visible(False)
        plt.axis("off")
        plt.tight_layout()
        for ap in self.animation_plots[:-1]:
            ap.set_data([], [])
        self.animation_plots[-1].set_text("t = 0.000")

        self.ee_poses = []
        self.tau_arrowarcs = []
        self.tau_arrowheads = []
        for link in range(self.plant.n_links):
            arc, head = get_arrow(
                radius=0.001, centX=0, centY=0, angle_=110, theta2_=320, color_="red"
            )
            self.tau_arrowarcs.append(arc)
            self.tau_arrowheads.append(head)
            self.animation_ax.add_patch(arc)
            self.animation_ax.add_patch(head)

        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def _animation_step(self, par_dict):
        """
        simulation of a single step which also updates the animation plot
        """
        dt = par_dict["dt"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        anim_dt = par_dict["anim_dt"]
        trail_len = 25  # length of the trails
        sim_steps = int(anim_dt / dt)

        realtime = True
        for _ in range(sim_steps):
            if controller is not None:
                t0 = time.time()
                tau = controller.get_control_output(x=self.x, t=self.t)
                if time.time() - t0 > dt:
                    realtime = False
            else:
                tau = np.zeros(self.plant.n_actuators)
            self.step(tau, dt, integrator=integrator)

        ani_plot_counter = 0

        # desired trajectory (shadow)
        if self.desired_state:
            ee_pos_des = self.plant.forward_kinematics(self.x_des[: self.plant.dof])
        if self.desired_traj:
            t = min(self.t, self.T_des[-2])
            ind = np.argwhere(self.T_des > t)[0]
            x = self.X_des[ind][0]
            ee_pos_des = self.plant.forward_kinematics(x[: self.plant.dof])

        if self.desired_state or self.desired_traj:
            ee_pos_des.insert(0, self.plant.base)
            for link in range(self.plant.n_links):
                self.animation_plots[ani_plot_counter].set_data(
                    [ee_pos_des[link][0], ee_pos_des[link + 1][0]],
                    [ee_pos_des[link][1], ee_pos_des[link + 1][1]],
                )
                ani_plot_counter += 1

        # regular pendulum
        ee_pos = self.plant.forward_kinematics(self.x[: self.plant.dof])
        ee_pos.insert(0, self.plant.base)

        self.ee_poses.append(ee_pos)
        if len(self.ee_poses) > trail_len:
            self.ee_poses = np.delete(self.ee_poses, 0, 0).tolist()

        # plot links
        for link in range(self.plant.n_links):
            self.animation_plots[ani_plot_counter].set_data(
                [ee_pos[link][0], ee_pos[link + 1][0]],
                [ee_pos[link][1], ee_pos[link + 1][1]],
            )
            ani_plot_counter += 1

        # plot base
        self.animation_plots[ani_plot_counter].set_data(ee_pos[0][0], ee_pos[0][1])
        ani_plot_counter += 1

        # desired trajectory (shadow)
        if self.desired_state or self.desired_traj:
            for link in range(self.plant.n_links):
                self.animation_plots[ani_plot_counter].set_data(
                    ee_pos_des[link + 1][0], ee_pos_des[link + 1][1]
                )
                ani_plot_counter += 1

                set_arrow_properties(
                    self.tau_arrowarcs[link],
                    self.tau_arrowheads[link],
                    tau[link],
                    ee_pos[link][0],
                    ee_pos[link][1],
                )

        # plot bodies
        for link in range(self.plant.n_links):
            self.animation_plots[ani_plot_counter].set_data(
                ee_pos[link + 1][0], ee_pos[link + 1][1]
            )
            ani_plot_counter += 1

            if self.plot_trail:
                self.animation_plots[ani_plot_counter].set_data(
                    np.asarray(self.ee_poses)[:, link + 1, 0],
                    np.asarray(self.ee_poses)[:, link + 1, 1],
                )
                ani_plot_counter += 1

            set_arrow_properties(
                self.tau_arrowarcs[link],
                self.tau_arrowheads[link],
                tau[link],
                ee_pos[link][0],
                ee_pos[link][1],
            )

        if self.plot_inittraj:
            T, X, U = controller.get_init_trajectory()
            coords = []
            for x in X:
                coords.append(self.plant.forward_kinematics(x[: self.plant.dof])[-1])

            coords = np.asarray(coords)
            self.animation_plots[ani_plot_counter].set_data(coords.T[0], coords.T[1])
            ani_plot_counter += 1

        if self.plot_forecast:
            T, X, U = controller.get_forecast()
            coords = []
            for x in X:
                coords.append(self.plant.forward_kinematics(x[: self.plant.dof])[-1])

            coords = np.asarray(coords)
            self.animation_plots[ani_plot_counter].set_data(coords.T[0], coords.T[1])
            ani_plot_counter += 1

        t = float(self.animation_plots[ani_plot_counter].get_text()[4:])
        t = round(t + dt * sim_steps, 3)
        self.animation_plots[ani_plot_counter].set_text(f"t = {t}")

        # if the animation runs slower than real time
        # the time display will be red
        if not realtime:
            self.animation_plots[ani_plot_counter].set_color("red")
        else:
            self.animation_plots[ani_plot_counter].set_color("black")

        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def simulate_and_animate(
        self,
        t0,
        x0,
        tf,
        dt,
        controller=None,
        integrator="runge_kutta",
        plot_inittraj=False,
        plot_forecast=False,
        plot_trail=True,
        phase_plot=False,
        save_video=False,
        video_name="pendulum_swingup",
        anim_dt=0.02,
    ):
        """
        Simulation and animation of the pendulum motion
        The animation is only implemented for 2d serial chains
        """

        self.plot_inittraj = plot_inittraj
        self.plot_forecast = plot_forecast
        self.plot_trail = plot_trail
        self.set_state(t0, x0)
        self.reset_data_recorder()

        fig = plt.figure(figsize=(5, 5))
        self.animation_ax = plt.axes()
        self.animation_plots = []

        colors = ["#0077BE", "#f66338"]
        colors_trails = ["#d2eeff", "#ffebd8"]

        if self.desired_state or self.desired_traj:
            for link in range(self.plant.n_links):
                (bar_plot,) = self.animation_ax.plot([], [], "-", lw=2, color="grey")
                self.animation_plots.append(bar_plot)

        for link in range(self.plant.n_links):
            (bar_plot,) = self.animation_ax.plot([], [], "-", lw=2, color="k")
            self.animation_plots.append(bar_plot)

        (base_plot,) = self.animation_ax.plot(
            [], [], "s", markersize=5.0, color="black"
        )
        self.animation_plots.append(base_plot)

        if self.desired_state or self.desired_traj:
            for link in range(self.plant.n_links):
                (ee_plot,) = self.animation_ax.plot(
                    [], [], "o", markersize=10.0, color="grey", markerfacecolor="grey"
                )
                self.animation_plots.append(ee_plot)

        for link in range(self.plant.n_links):
            (ee_plot,) = self.animation_ax.plot(
                [],
                [],
                "o",
                markersize=10.0,
                color="k",
                markerfacecolor=colors[link % len(colors)],
            )
            self.animation_plots.append(ee_plot)

            if self.plot_trail:
                (trail_plot,) = self.animation_ax.plot(
                    [],
                    [],
                    "-",
                    color=colors[link],
                    markersize=6,
                    markerfacecolor=colors_trails[link % len(colors_trails)],
                    lw=2,
                    markevery=10000,
                    markeredgecolor="None",
                )
                self.animation_plots.append(trail_plot)

        if self.plot_inittraj:
            (it_plot,) = self.animation_ax.plot([], [], "--", lw=1, color="gray")
            self.animation_plots.append(it_plot)
        if self.plot_forecast:
            (fc_plot,) = self.animation_ax.plot([], [], "-", lw=1, color="green")
            self.animation_plots.append(fc_plot)

        text_plot = self.animation_ax.text(
            0.1, 0.9, [], fontsize=20, transform=fig.transFigure
        )

        self.animation_plots.append(text_plot)

        num_steps = int(tf / anim_dt)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["anim_dt"] = anim_dt
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        frames = num_steps * [par_dict]

        animation = FuncAnimation(
            fig,
            self._animation_step,
            frames=frames,
            init_func=self._animation_init,
            blit=True,
            repeat=False,
            interval=dt * 1000,
            cache_frame_data=False,
        )

        if save_video:
            print(f"Saving video to {video_name}.mp4")
            Writer = mplanimation.writers["ffmpeg"]
            writer = Writer(fps=60, bitrate=1800)
            animation.save(video_name + ".mp4", writer=writer)
            print("Saving video done.")
        else:
            self.set_state(t0, x0)
            self.reset_data_recorder()
            #plt.show()
        #plt.close()

        return self.t_values, self.x_values, self.tau_values, animation

    def set_desired_traj(self, T, X):
        self.T_des = np.asarray(T)
        self.X_des = np.asarray(X)
        self.desired_traj = True

    def set_desired_state(self, x):
        self.x_des = np.asarray(x)
        self.desired_state = True
        
    def activate_hardware(self):
        """
        Activate the pendulum hardware
        """   
        import pyCandle
        # Create CANdle object and set FDCAN baudrate to 1Mbps
        self.candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M,True)
        # Ping FDCAN bus in search of drives
        ids = self.candle.ping()
        # Add all found to the update list
        for id in ids:
            self.candle.addMd80(id)
            
    def run_on_hardware(self, tf, dt, controller=None):
        import pyCandle
        import time

        # Select pendulum from motor list

        # Now we shall loop over all found drives to change control mode and enable them one by one
        for md in self.candle.md80s:
            self.candle.controlMd80SetEncoderZero(md)      #  Reset encoder at current position
            self.candle.controlMd80Mode(md, pyCandle.IMPEDANCE)    # Set mode to impedance control
            self.candle.controlMd80Enable(md, True)     # Enable the drive

        # Begin update loop (it starts in the background)

        self.candle.begin()
        candle_dict = {}

        motornum = 0

        for motor in self.candle.md80s:
            candle_dict[self.candle.md80s[motornum].getId()] = motornum
            motornum += 1

        shoulder_motor_id = 171
        elbow_motor_id = 172
        
        shoulder_motor = candle_dict[shoulder_motor_id]
        elbow_motor = candle_dict[elbow_motor_id]
        
        # set zero impedance (kp=kd=0) for pure torque control
        self.candle.md80s[shoulder_motor].setImpedanceControllerParams(0, 0)
        self.candle.md80s[elbow_motor].setImpedanceControllerParams(0, 0)
        
        input("Press bring the pendulum to the starting configuration and press enter to continue...")
        n = int(tf / dt)

        meas_time_vec = np.zeros(n)
        meas_pos_shoulder = np.zeros(n)
        meas_vel_shoulder = np.zeros(n)
        meas_tau_shoulder = np.zeros(n)
        des_tau_shoulder = np.zeros(n)
        meas_pos_elbow = np.zeros(n)
        meas_vel_elbow = np.zeros(n)
        meas_tau_elbow = np.zeros(n)
        des_tau_elbow = np.zeros(n)
        
        tau = [0.0, 0.0]
        
        # defining runtime variables
        i = 0
        meas_dt = 0.0
        meas_time = 0.0
        print("Control Loop Started!")
        
        safety_position_limit = 4.0 * np.pi  # example limits, replace with your values
        safety_velocity_limit = 50

        # Auto update loop is running in the background updating data in candle.md80s vector. Each md80 object can be
        # Called for data at any time
        while i < n:
            start_loop = time.time()
            meas_time += meas_dt

            ## Do your stuff here - START
            measured_position_shoulder = self.candle.md80s[shoulder_motor].getPosition()
            measured_velocity_shoulder = self.candle.md80s[shoulder_motor].getVelocity() 
            measured_torque_shoulder = self.candle.md80s[shoulder_motor].getTorque()   
            measured_position_elbow = self.candle.md80s[elbow_motor].getPosition()
            measured_velocity_elbow = self.candle.md80s[elbow_motor].getVelocity() 
            measured_torque_elbow = self.candle.md80s[elbow_motor].getTorque()
             # Check safety conditions
            if (
                abs(measured_position_shoulder) > safety_position_limit
                or abs(measured_position_elbow) > safety_position_limit
                or abs(measured_velocity_shoulder) > safety_velocity_limit
                or abs(measured_velocity_elbow) > safety_velocity_limit
            ):
                print("Safety conditions violated! Slowing down motors for 1 second.")
                # Slow down motors with kd=1 for 1 second at current rate dt
                steps = int(1 / dt)
                for _ in range(steps):
                    # send impedance command with kd=1 (derivative gain) to slow down motors
                    self.candle.md80s[shoulder_motor].setImpedanceControllerParams(
                        0, 0.1
                    )
                    self.candle.md80s[elbow_motor].setImpedanceControllerParams(0, 0.1)

                    # Zero torque to let the impedance damping slow the motor
                    self.candle.md80s[shoulder_motor].setTargetTorque(0)
                    self.candle.md80s[elbow_motor].setTargetTorque(0)

                    time.sleep(dt)

                print(
                    f"The limit violating state was ({measured_position_shoulder}, {measured_position_elbow}, {measured_velocity_shoulder}, {measured_velocity_elbow})"
                )
                break  
            
            self.x = np.array([measured_position_shoulder, measured_position_elbow, measured_velocity_shoulder, measured_velocity_elbow])
            
            # Control logic
            if controller is not None:
                tau = controller.get_control_output(self.x)
                self.candle.md80s[shoulder_motor].setTargetTorque(tau[0])
                self.candle.md80s[elbow_motor].setTargetTorque(tau[1])
            else:
                tau[0] = 0               
                tau[1] = 0               

            # Collect data for plotting
            meas_time_vec[i] = meas_time
            meas_pos_shoulder[i] = measured_position_shoulder
            meas_vel_shoulder[i] = measured_velocity_shoulder   
            meas_tau_shoulder[i] = measured_torque_shoulder
            des_tau_shoulder[i] = tau[0]
            meas_pos_elbow[i] = measured_position_elbow
            meas_vel_elbow[i] = measured_velocity_elbow   
            meas_tau_elbow[i] = measured_torque_elbow
            des_tau_elbow[i] = tau[1]
            
            ## Do your stuff here - END
            i += 1
            exec_time = time.time() - start_loop
            if exec_time > dt:
                print("Control loop is too slow!")
                print("Control frequency:", 1/exec_time, "Hz")
                print("Desired frequency:", 1/dt, "Hz")
                print()

            while time.time() - start_loop < dt:
                pass
            meas_dt = time.time() - start_loop
        print("Control Loop Ended!")

        # Send a few zeros to the motor and then close the update loop
        for i in range(5):
            self.candle.md80s[shoulder_motor].setTargetTorque(0.0)
            self.candle.md80s[elbow_motor].setTargetTorque(0.0)
        self.candle.end()

        self.t_values = meas_time_vec
        self.x_values = np.vstack((meas_pos_shoulder, meas_pos_elbow, meas_vel_shoulder, meas_vel_elbow)).T
        self.tau_values = np.vstack((meas_tau_shoulder, meas_tau_elbow)).T
        self.des_tau_values = np.vstack((des_tau_shoulder, des_tau_elbow)).T

        return self.t_values, self.x_values, self.tau_values, self.des_tau_values

