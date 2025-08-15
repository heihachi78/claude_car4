import matplotlib.pyplot as plt
import math



class SimplePhysics():

    def __init__(self, car_mass = 1500, #1500
                 car_horsepower = 650, #650
                 car_air_resistance = 0.325, #0.325
                 car_downforce = 0.35, #0.35
                 car_torque = 800, #800
                 car_brake_power = 900, #900
                 car_suspension_stiffness = 100, #100
                 car_max_steering_angle_deg = 15, #15
                 car_wheelbase = 4, #4
                 car_antiroll_bar = 100, #100
                 car_weight_distribution_front = 0.5, #0.5
                 car_brake_bias_front = 0.6, #0.6
                 tyre_rolling_resistance=0.015, #0.015
                 tyre_friction_coefficient = 1.2, #1.2
                 sim_per_sec=60, #60
                 ): 
        self.acceleration_throttle_hist = []
        self.current_speed_mps_hist = []
        self.air_resistance_hist = []
        self.power_accel_hist = []
        self.resist_accel_hist = []
        self.braking_force_hist = []
        self.deceleration_brake_hist = []
        self.drag_load_hist = []
        self.tyre_load_fr_hist = []
        self.tyre_load_fl_hist = []
        self.tyre_load_rr_hist = []
        self.tyre_load_rl_hist = []
        self.roll_force_hist = []
        self.acceleration_steering_hist = []
        self.x_hist = []
        self.y_hist = []

        self.CAR_MASS = car_mass
        self.CAR_HORSEPOWER = car_horsepower
        self.CAR_AIR_RESISTANCE = car_air_resistance
        self.CAR_DOWNFORCE = car_downforce
        self.CAR_TORQUE = car_torque
        self.CAR_ACCELERATION_COEFFICIENT = 0.03 #0.03
        self.CAR_TOP_SPEED_COEFFICIENT = 4 #4
        self.CAR_BRAKE_POWER = car_brake_power / 45 #45
        self.CAR_BRAKE_EFFICIENCY_COEFFICIENT = 8 #8
        self.CAR_SUSPENSION_STIFFNESS = car_suspension_stiffness
        self.CAR_MAX_STEERING_ANGLE_DEG = car_max_steering_angle_deg
        self.CAR_WHEELBASE = car_wheelbase
        self.CAR_ANTIROLL_BAR = car_antiroll_bar
        self.CAR_WEIGHT_DISTRIBUTION_FRONT = car_weight_distribution_front
        self.CAR_WEIGHT_DISTRIBUTION_REAR = 1 - car_weight_distribution_front
        self.CAR_BRAKE_BIAS_FRONT = car_brake_bias_front
        self.CAR_BRAKE_BIAS_REAR = 1 - car_brake_bias_front
        self.TYRE_FRICTION_COEFFICIENT = tyre_friction_coefficient

        self.TYRE_ROLLING_RESISTANCE = tyre_rolling_resistance

        self.G = 9.81 #9.81
        self.AIR_DENSITY = 1.225 #1.225

        self.SIM_PER_SEC = sim_per_sec

        self.total_air_resistance = 0
        self._calculate_total_air_resistance()
        self.car_power_on_wheels = 0
        self._calculate_power_on_wheels()
        self.theoretical_top_speed_mps = 0
        self._calculate_theoretical_top_speed()

        self.acceleration_throttle = 0
        self.deceleration_brake = 0
        self.acceleration_steering = 0
        self.current_speed_mps = 0
        self.current_steering_angle_deg = 0
        self.x = 0.0
        self.y = 0.0
        self.heading_rad = 0.0  # 0 rad = facing along +X


        self.tyre_load_fr = self.CAR_MASS / 4
        self.tyre_load_fl = self.CAR_MASS / 4
        self.tyre_load_rr = self.CAR_MASS / 4
        self.tyre_load_rl = self.CAR_MASS / 4

    def _reset_hist(self):
        self.acceleration_throttle_hist = []
        self.current_speed_mps_hist = []
        self.air_resistance_hist = []
        self.power_accel_hist = []
        self.resist_accel_hist = []
        self.braking_force_hist = []
        self.deceleration_brake_hist = []
        self.drag_load_hist = []
        self.tyre_load_fr_hist = []
        self.tyre_load_fl_hist = []
        self.tyre_load_rr_hist = []
        self.tyre_load_rl_hist = []
        self.roll_force_hist = []
        self.acceleration_steering_hist = []
        self.x_hist = []
        self.y_hist = []

    def show_deceleration_test_hist(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.air_resistance_hist, label='air_resistance_hist')
        axs[0, 0].set_title("air_resistance_hist")
        axs[1, 0].plot(self.braking_force_hist, label='braking_force_hist')
        axs[1, 0].set_title("braking_force_hist")
        axs[1, 1].plot(self.current_speed_mps_hist, label='current_speed_mps_hist')
        axs[1, 1].set_title("current_speed_mps_hist")
        axs[0, 1].plot(self.deceleration_brake_hist, label='self.deceleration_brake_hist')
        axs[0, 1].set_title("self.deceleration_brake_hist")
        fig.tight_layout()
        plt.show()

    def show_acceleration_test_hist(self):
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].plot(self.acceleration_throttle_hist, label='acceleration_throttle_hist')
        axs[0, 0].set_title("acceleration_throttle_hist")
        axs[1, 0].plot(self.current_speed_mps_hist, label='current_speed_mps_hist')
        axs[1, 0].set_title("current_speed_mps_hist")
        axs[0, 1].plot(self.air_resistance_hist, label='air_resistance_hist')
        axs[0, 1].set_title("air_resistance_hist")
        axs[1, 1].plot(self.power_accel_hist, label='power_accel_hist')
        axs[1, 1].set_title("power_accel_hist")
        axs[1, 2].plot(self.resist_accel_hist, label='resist_accel_hist')
        axs[1, 2].set_title("resist_accel_hist")
        fig.tight_layout()
        plt.show()

    def show_info(self):
        print('='*20, ' INFOBOX ', '='*20)
        print(f'theoretical_top_speed_mps; {self.theoretical_top_speed_mps:.2f}, theoretical_top_speed_kph: {(self.theoretical_top_speed_mps*3.6):.2f}')
        print(f'total_air_resistance: {self.total_air_resistance:.2f}')
        print(f'power_on_wheels: {self.car_power_on_wheels:.2f}')
        print('='*20, '=========', '='*20)

    def show_throttle_times(self):
        kmph_limit = 50
        kmph_step = 50
        for i, kmph in enumerate([x * 3.6 for x in scp.current_speed_mps_hist]):
            if kmph > kmph_limit:
                print(f'{kmph_limit}: {(i/scp.SIM_PER_SEC):.2f}')
                kmph_limit += kmph_step
        print(f'top speed: {3.6 * scp.current_speed_mps_hist[-1]:.2f}')

    def show_brake_times(self):
        kmph_limit = 0
        kmph_step = 50
        zero_limit = 0
        for i, kmph in enumerate(reversed([x * 3.6 for x in scp.current_speed_mps_hist])):
            if kmph > kmph_limit:
                if zero_limit > 0:
                    print(f'{kmph_limit}: {((i/scp.SIM_PER_SEC)-zero_limit):.2f}')
                kmph_limit += kmph_step
                if zero_limit == 0:
                    zero_limit = i/scp.SIM_PER_SEC
        print(f'top speed: {3.6 * scp.current_speed_mps_hist[0]:.2f}')

    def show_tyre_load_hist(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 1].plot(self.tyre_load_fr_hist, label='tyre_load_fr_hist')
        axs[0, 1].set_title("tyre_load_fr_hist")
        axs[0, 0].plot(self.tyre_load_fl_hist, label='tyre_load_fl_hist')
        axs[0, 0].set_title("tyre_load_fl_hist")
        axs[1, 1].plot(self.tyre_load_rr_hist, label='tyre_load_rr_hist')
        axs[1, 1].set_title("tyre_load_rr_hist")
        axs[1, 0].plot(self.tyre_load_rl_hist, label='tyre_load_rl_hist')
        axs[1, 0].set_title("tyre_load_rl_hist")
        fig.tight_layout()
        plt.show()

    def show_weight_transfer_hist(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 1].plot(self.drag_load_hist, label='drag_load_hist')
        axs[0, 1].set_title("drag_load_hist")
        axs[1, 0].plot(self.roll_force_hist, label='roll_force_hist')
        axs[1, 0].set_title("roll_force_hist")
        fig.tight_layout()
        plt.show()

    def show_steering_hist(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 1].plot(self.acceleration_steering_hist, label='acceleration_steering_hist')
        axs[0, 1].set_title("acceleration_steering_hist")
        fig.tight_layout()
        plt.show()

    def show_path(self):
        plt.figure()
        plt.plot(self.x_hist, self.y_hist)
        plt.axis('equal')
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.title("Car Path")
        plt.show()

    def _calculate_power_on_wheels(self):
        self.car_power_on_wheels = self.CAR_HORSEPOWER * 745.7

    def _calculate_total_air_resistance(self):
        self.total_air_resistance = self.CAR_AIR_RESISTANCE + self.CAR_DOWNFORCE

    def _calculate_theoretical_top_speed(self):
        def power_required(v):
            drag = self._get_air_resistance(v)
            roll = self.CAR_MASS * self.G * self.TYRE_ROLLING_RESISTANCE
            return (drag + roll) * v

        v = 0.0
        step = 0.1
        while power_required(v) < self.car_power_on_wheels:
            v += step

        self.theoretical_top_speed_mps = v

    def _get_cosinus_scaled(self, max, rate, pow):
        cto = math.pi/4
        chwere = cto * rate
        cval = math.cos(chwere)
        return cval**pow * max

    def _get_air_resistance(self, speed_mps):
        return 0.5 * self.AIR_DENSITY * self.total_air_resistance * speed_mps**2

    def _get_total_car_mass(self):
        return self.tyre_load_fr + self.tyre_load_fl + self.tyre_load_rr + self.tyre_load_rl

    def _get_roll_force(self):
        return self._get_total_car_mass() * self.G * self.TYRE_ROLLING_RESISTANCE

    def _weight_transfer(self):
        drag_load = self._get_air_resistance(self.current_speed_mps) / 4 * self.CAR_DOWNFORCE
        weight_load = self.CAR_MASS / 4
        self.drag_load_hist.append(drag_load)
        transfer_rate_acceleration = self.acceleration_throttle / self.CAR_SUSPENSION_STIFFNESS
        transfer_rate_deceleration = self.deceleration_brake / self.CAR_SUSPENSION_STIFFNESS
        transfer_rate_steering = self.acceleration_steering / self.CAR_ANTIROLL_BAR
        transfered_weight_to_front = -transfer_rate_deceleration * (self.tyre_load_rl + self.tyre_load_rr)
        transfered_weight_to_rear = transfer_rate_acceleration * (self.tyre_load_fl + self.tyre_load_fr)
        transfered_weight_to_left = transfer_rate_steering * (self.tyre_load_fr + self.tyre_load_rr)
        transfered_weight_to_right = -transfer_rate_steering * (self.tyre_load_rl + self.tyre_load_fl)
        calculated_tyre_load_fr = weight_load + drag_load + transfered_weight_to_front - transfered_weight_to_rear - transfered_weight_to_left + transfered_weight_to_right
        calculated_tyre_load_fl = weight_load + drag_load + transfered_weight_to_front - transfered_weight_to_rear + transfered_weight_to_left - transfered_weight_to_right
        calculated_tyre_load_rr = weight_load + drag_load - transfered_weight_to_front + transfered_weight_to_rear - transfered_weight_to_left + transfered_weight_to_right
        calculated_tyre_load_rl = weight_load + drag_load - transfered_weight_to_front + transfered_weight_to_rear + transfered_weight_to_left - transfered_weight_to_right
        self.tyre_load_fr = calculated_tyre_load_fr
        self.tyre_load_fl = calculated_tyre_load_fl
        self.tyre_load_rr = calculated_tyre_load_rr
        self.tyre_load_rl = calculated_tyre_load_rl
        self.tyre_load_fr_hist.append(self.tyre_load_fr)
        self.tyre_load_fl_hist.append(self.tyre_load_fl)
        self.tyre_load_rr_hist.append(self.tyre_load_rr)
        self.tyre_load_rl_hist.append(self.tyre_load_rl)

#    def _get_throttle_acceleration(self, throttle):
#        drag_force = self._get_air_resistance(self.current_speed_mps)
#        resist_accel = (drag_force + self._get_roll_force()) / self._get_total_car_mass()
#        power_accel = throttle * self._get_cosinus_scaled(max=self.CAR_TORQUE / self._get_total_car_mass() / self.CAR_ACCELERATION_COEFFICIENT, 
#                                                          rate=self.current_speed_mps / self.theoretical_top_speed_mps, 
#                                                          pow=self.CAR_TOP_SPEED_COEFFICIENT) 
#        max_accel = power_accel - resist_accel
#        self.power_accel_hist.append(power_accel)
#        self.resist_accel_hist.append(resist_accel)
#        self.air_resistance_hist.append(drag_force)
#        self.roll_force_hist.append(self._get_roll_force())
#        return max(0.0, max_accel)

    def _get_throttle_acceleration(self, throttle):
        # Replace the complex cosine function with a simpler curve
        speed_ratio = self.current_speed_mps / self.theoretical_top_speed_mps
        power_multiplier = max(0, 1 - speed_ratio**2)  # Simple quadratic falloff
        
        max_force = self.CAR_TORQUE * throttle * power_multiplier
        drag_force = self._get_air_resistance(self.current_speed_mps)
        roll_force = self._get_roll_force()
        
        net_force = max_force - drag_force - roll_force
        return max(0, net_force / self._get_total_car_mass())

    def _get_break_acceleration(self, brake):
        braking_force = brake * self._get_cosinus_scaled(max=self.CAR_BRAKE_POWER * self._get_total_car_mass(), 
                                                         rate=self.current_speed_mps / self.theoretical_top_speed_mps, 
                                                         pow=self.CAR_BRAKE_EFFICIENCY_COEFFICIENT)
        drag_force = self._get_air_resistance(self.current_speed_mps)
        decel = (braking_force + drag_force + self._get_roll_force()) / self._get_total_car_mass()
        self.braking_force_hist.append(braking_force)
        self.air_resistance_hist.append(drag_force)
        self.roll_force_hist.append(self._get_roll_force())
        if self.current_speed_mps <= 0:
            decel = 0
            self.current_speed_mps = 0
        return min(-decel, 0)
    
    def _get_steering_acceleration(self, steering):
        self.current_steering_angle_deg = steering * self.CAR_MAX_STEERING_ANGLE_DEG
        if abs(self.current_steering_angle_deg) > 0.001:
            turning_radius = self.CAR_WHEELBASE / math.sin(math.radians(self.current_steering_angle_deg))
            angular_velocity_deg_per_sec = (self.current_speed_mps / turning_radius) * (180 / math.pi)
        else:
            angular_velocity_deg_per_sec = 0.0
        return angular_velocity_deg_per_sec / self.CAR_ANTIROLL_BAR

    def _throttle(self, throttle):
        self.acceleration_throttle = self._get_throttle_acceleration(throttle)
        #self.current_speed_mps += self.acceleration_throttle * 1/self.SIM_PER_SEC
        self.acceleration_throttle_hist.append(self.acceleration_throttle)
        self.current_speed_mps_hist.append(self.current_speed_mps)

    def _brake(self, brake):
        self.deceleration_brake = self._get_break_acceleration(brake)
        #self.current_speed_mps += self.deceleration_brake * 1/self.SIM_PER_SEC
        self.deceleration_brake_hist.append(self.deceleration_brake)
        self.current_speed_mps_hist.append(self.current_speed_mps)

    def _steering(self, steering):
        self.acceleration_steering = self._get_steering_acceleration(steering)
        #self.acceleration_steering_hist.append(self.acceleration_steering)

    def drive(self, throttle, brake, steering):
        self._throttle(throttle)
        self._brake(brake)
        self._steering(steering)
        self._weight_transfer()

        longitudinal_force_front = 0.0
        longitudinal_force_rear = 0.0

        if abs(self.current_steering_angle_deg) > 0.001:
            turning_radius = self.CAR_WHEELBASE / math.sin(math.radians(self.current_steering_angle_deg))
            total_lateral_force = self.CAR_MASS * (self.current_speed_mps**2 / turning_radius)
            lateral_force_front = total_lateral_force * self.CAR_WEIGHT_DISTRIBUTION_FRONT
            lateral_force_rear = total_lateral_force * self.CAR_WEIGHT_DISTRIBUTION_REAR
        else:
            lateral_force_front = 0.0
            lateral_force_rear = 0.0

        if self.acceleration_throttle > 0:
            longitudinal_force_front = 0.0
            longitudinal_force_rear = self.CAR_MASS * self.acceleration_throttle

        if self.deceleration_brake < 0:
            brake_force_total = self.CAR_MASS * self.deceleration_brake
            longitudinal_force_front += brake_force_total * self.CAR_BRAKE_BIAS_FRONT
            longitudinal_force_rear += brake_force_total * self.CAR_BRAKE_BIAS_REAR

        max_force_front = (self.tyre_load_fr + self.tyre_load_fl) * self.G * self.TYRE_FRICTION_COEFFICIENT
        max_force_rear = (self.tyre_load_rr + self.tyre_load_rl) * self.G * self.TYRE_FRICTION_COEFFICIENT

        total_force_front = math.sqrt(longitudinal_force_front**2 + lateral_force_front**2)
        if total_force_front > max_force_front:
            scale = max_force_front / total_force_front
            longitudinal_force_front *= scale
            lateral_force_front *= scale

        total_force_rear = math.sqrt(longitudinal_force_rear**2 + lateral_force_rear**2)
        if total_force_rear > max_force_rear:
            scale = max_force_rear / total_force_rear
            longitudinal_force_rear *= scale
            lateral_force_rear *= scale

        net_longitudinal_force = longitudinal_force_front + longitudinal_force_rear
        net_longitudinal_accel = net_longitudinal_force / self.CAR_MASS
        net_lateral_force = lateral_force_front + lateral_force_rear
        net_lateral_accel = net_lateral_force / self.CAR_MASS

        self.acceleration_steering = net_lateral_accel
        self.current_speed_mps += net_longitudinal_accel * (1 / self.SIM_PER_SEC)
        self.acceleration_steering_hist.append(self.acceleration_steering)

        if abs(self.current_steering_angle_deg) > 0.001:
            turning_radius = self.CAR_WHEELBASE / math.sin(math.radians(self.current_steering_angle_deg))
            angular_velocity = self.current_speed_mps / turning_radius
        else:
            angular_velocity = 0.0

        self.heading_rad += angular_velocity * (1 / self.SIM_PER_SEC)
        self.heading_rad = (self.heading_rad + math.pi) % (2 * math.pi) - math.pi
        self.x += self.current_speed_mps * math.cos(self.heading_rad) * (1 / self.SIM_PER_SEC)
        self.y += self.current_speed_mps * math.sin(self.heading_rad) * (1 / self.SIM_PER_SEC)

        self.x_hist.append(self.x)
        self.y_hist.append(self.y)

def test_throttle_sec(scp: SimplePhysics, sec=60):
    print('='*20, ' THROTTLE ', '*'*20)
    scp._reset_hist()
    scp.show_info()
    for i in range(scp.SIM_PER_SEC * sec):
        scp.drive(1, 0, 1)
    scp.show_throttle_times()
    scp.show_acceleration_test_hist()
    scp.show_weight_transfer_hist()
    scp.show_tyre_load_hist()
    scp.show_steering_hist()

def test_brake_sec(scp: SimplePhysics, sec=60):
    print('='*20, '   BRAKE  ', '*'*20)
    #scp._reset_hist()
    scp.show_info()
    for i in range(scp.SIM_PER_SEC * sec):
        scp.drive(0, 1, -1)
    scp.show_brake_times()
    scp.show_deceleration_test_hist()
    scp.show_weight_transfer_hist()
    scp.show_tyre_load_hist()
    scp.show_steering_hist()

if __name__ == '__main__':
    scp = SimplePhysics()
    test_throttle_sec(scp, 60)
    test_brake_sec(scp, 60)
    scp.show_path()
