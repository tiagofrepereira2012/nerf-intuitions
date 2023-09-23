from pathlib import Path
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np

from volumetric_rendering import rendering


# FOr loading purposes
from voxel_reconstruction import get_rays, Voxels
from nerf import NeRF

import mcubes
import trimesh

def main():

    parser = argparse.ArgumentParser(
        description="Reconstruct the 3D scene from the model"
    )
    parser.add_argument("model_path", type=Path, help="Path to the model")

    args = parser.parse_args()

    model_path = args.model_path

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    model = torch.load(model_path).to(device)

    N = 100
    tn = 8
    tf = 12
    nb_bins = 100
    scale = 1.5

    x= torch.linspace(-scale, scale, N)
    y= torch.linspace(-scale, scale, N)
    z= torch.linspace(-scale, scale, N)


    x, y, z = torch.meshgrid(x, y, z)

    xyz = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)

    # The goal is to query the NeRF model to get the density of a particular voxel
    # Then we will render this density map

    with torch.no_grad():
        _, density = model.forward(xyz.to(device), torch.zeros_like(xyz).to(device))
    
    # Generating the triangles given the density
    density = density.cpu().numpy().reshape(N, N, N)
    vertices, triangles = mcubes.marching_cubes(density, 30*np.mean(density))

    # Lets printing he mesh

    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.show()

    pass

    

if __name__ == "__main__":
    main()