# Honk Sound Wavefront Class
# Tung M. Phan
# California Institute of Technology
# July 19, 2018

import numpy as np
speed_of_sound = 500 # speed of sound constant for wave propagation calculation

class HonkWavefront:
    """HonkWavefront Class

    init_state is [x, y, r, I], where x, y, r, I are the positions, radius, and intensity of the wavefront
    init_energy is the initial energy of the burst
    """
    def __init__(self, init_state = [0, 0, 0, 1], init_energy = 1000): # x, y, radius, intensity
                     self.state = np.array(init_state, dtype='float')
                     self.energy = init_energy

    def next(self,dt):
       """
       update the wave_front

       """
       self.state[2] = self.state[2] + dt * speed_of_sound
       if self.state[2] == 0:
           self.state[3] = 1
       else:
           self.state[3] = min(1, max(0, self.energy / (2 * np.pi * self.state[2])))

    def get_data(self):
       """
       return plotting data

       """
       return self.state
