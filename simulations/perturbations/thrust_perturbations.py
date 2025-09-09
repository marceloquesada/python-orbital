from utils import orbitalElements, refSystems
import numpy as np

orbitalElements.get_orbital_elements

class FixedMassThetaIntervalThrust:
    def __init__(self, thrust, thrust_direction, m_sat, mu, angle_intervals):
        self.thrust = thrust
        self.thrust_direction = thrust_direction
        self.m_sat = m_sat
        self.mu = mu
        self.angle_intervals = angle_intervals

    def get_acceleration(self, state_vector, dt):
        theta = orbitalElements.get_true_anomaly(state_vector[0:3], state_vector[3:6], self.mu)
        mass = self.m_sat
        angle_intervals = self.angle_intervals

        if any(lim1 <= theta <= lim2 for (lim1, lim2) in angle_intervals):
            base = refSystems.get_base_versors(state_vector)
            inertial_thrust_force_vector = self.thrust*(base @ self.thrust_direction)
            a_thrust = inertial_thrust_force_vector/mass
        else:
            a_thrust = 0
        
        return a_thrust


class VariableMassThetaIntervalThrust:
    def __init__(self, thrust, thrust_direction, m_dry, m_prop, angle_intervals: list[tuple], Isp: float = None, mdot: float = None):
        self.thrust = thrust
        self.thrust_direction = thrust_direction
        self.m_dry = m_dry
        self.m_prop_0 = m_prop
        self.m_prop = m_prop
        self.angle_intervals = angle_intervals

        if Isp is None:
            if mdot is None:
                raise Exception("Either the Isp or mdot for the thruster has to be given")
            else:
                self.mdot = mdot
        else:
            self.mdot = thrust/Isp

    def get_acceleration(self, state_vector, dt):
        theta = orbitalElements.get_true_anomaly(state_vector[0:3], state_vector[3:6], self.mu)
        mass = self.m_dry + self.m_prop
        angle_intervals = self.angle_intervals

        if self.m_prop > 0:
            if any(lim1 <= theta <= lim2 for (lim1, lim2) in angle_intervals):
                base = refSystems.get_base_versors(state_vector)
                inertial_thrust_force_vector = self.thrust*(base @ self.thrust_direction)
                a_thrust = inertial_thrust_force_vector/mass
                self.m_prop = self.m_prop - self.mdot*dt
            else:
                a_thrust = 0
        else:
            a_thrust = 0
        
        return a_thrust