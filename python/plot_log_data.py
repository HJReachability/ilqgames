import numpy as np
from plotter import *

if __name__ == "__main__":
    #ilqg_filename = "./logs/unicycle_4d_example.pkl"
    ilqg_filename = "./logs/two_player_zero_sum/goal_75_100_init_pi_6_v_5_dist.pkl"
    hji_filename = "./logs/unicycle_4d_example_hji.mat"
    plotter = Plotter(ilqg_filename, hji_filename)
    plotter.plot_controls()
    plotter.plot_disturbances()
    plotter.plot_player_costs()
    plotter.plot_trajectories()


    
