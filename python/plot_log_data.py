import numpy as np
from plotter import *

if __name__ == "__main__":
    ilqg_filename = "./logs/unicycle_4d_example.pkl"
    hji_filename = "./logs/unicycle_4d_example_hji.mat"
    plotter = Plotter(ilqg_filename, hji_filename)
    plotter.plot_controls()
    plotter.plot_disturbances()
    plotter.plot_player_costs()
    plotter.plot_trajectories()


    
