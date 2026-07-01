import numpy as np
import matplotlib.pyplot as plt


class MemristiveSynapse:
    def __init__(
        self,
        G_init=0.5,
        G_min=0.0,
        G_max=1.0,
        A_plus=0.03,
        A_minus=0.025,
        tau_plus=20.0,
        tau_minus=20.0,
        noise_std=0.002,
    ):
        self.G = G_init
        self.G_min = G_min
        self.G_max = G_max
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.noise_std = noise_std

        self.last_pre_t = None
        self.last_post_t = None

    def current(self, pre_spike):
        """
        pre_spike가 들어오면 conductance G에 비례해서 current 발생.
        """
        return self.G * pre_spike
    
    def stdp_delta(self, dt):
        """
        dt = t_post - t_pre
        dt > 0: pre before post -> LTP
        dt < 0: post before pre -> LTD
        dt = 0: ambiguous -> no update
        """
        if dt > 0:
            return self.A_plus * np.exp(-dt / self.tau_plus)
        elif dt < 0:
            return -self.A_minus * np.exp(dt / self.tau_minus)
        else:
            return 0.0
        
    def update_pre(self, t):
        """
        pre-spike 발생 시점에서 STDP depression 또는 potentiation 적용.
        """
        self.last_pre_t = t

        if self.last_post_t is not None:
            dt = self.last_post_t - self.last_pre_t

            # post가 pre보다 먼저 발생한 경우: depression
            if dt < 0:
                dG = self.stdp_delta(dt)
                self.apply_update(dG)

    def update_post(self, t):
        """
        post-spike 발생 시점에서 STDP potentiation 적용.
        """
        self.last_post_t = t

        if self.last_pre_t is not None:
            dt = self.last_post_t - self.last_pre_t

            # pre가 post보다 먼저 발생한 경우: potentiation
            if dt > 0:
                dG = self.stdp_delta(dt)
                self.apply_update(dG)

    def apply_update(self, dG):
        """
        memristor-like non-ideality:
        - conductance range 제한
        - update noise
        - conductance가 max/min에 가까울수록 update 작아지는 saturation
        """
        if dG > 0:
            saturation = 1.0 - (self.G - self.G_min) / (self.G_max - self.G_min)
        else:
            saturation = (self.G - self.G_min) / (self.G_max - self.G_min)

        noise = np.random.normal(0, self.noise_std)
        self.G += dG * saturation + noise
        self.G = np.clip(self.G, self.G_min, self.G_max)
