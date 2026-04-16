import h5py
import numpy as np

with h5py.File("pick_egg_safe_combined.hdf5", "r") as f:
    def print_structure(name, obj):
        print(name, "->", obj)
    
    f.visititems(print_structure)