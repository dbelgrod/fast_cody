import os
import numpy as np
import scipy as sp
import igl
import json
from os.path import basename, splitext

import fast_cody as fc

def contained_cd_affine_handle(msh_file=None, texture_png=None, texture_obj=None):
    """
    Runs a standard interactive fast CD simulation, where the user can manipulate a single affine
    handle with a Guizmo and observe secondary effects in real-time.
    """
    if msh_file is not None:
        [V, F, T] = fc.read_msh(msh_file)
    elif msh_file is None:
        msh_file = fc.get_data("./cd_fish.msh")
        [V, F, T] = fc.read_msh(msh_file)

    if texture_png is None or texture_obj is None:
        if msh_file ==  fc.get_data("./cd_fish.msh"):
            texture_png = fc.get_data("./cd_fish_tex.png")
            texture_obj = fc.get_data("./cd_fish_tex.obj")

    #TODO: Scale and center geometry
    
    #TODO: Linear Blend Skinning Jacobian

    #TODO: Complementary Constraint Matrix

    #TODO: LBS Weight Space Constraint

    #TODO: Skinning Subspace
    
    #TODO: Fast CD Sim

    #TODO: Set sim state and initial rig parameters

    #TODO: Pre-draw callback
    
    