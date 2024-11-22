import os, sys; os.chdir(f"{os.environ['HOME']}/01_repos")
sys.path.append(os.getcwd())

from paths import Paths
import mlflow
from mlflow.tracking import MlflowClient

from tqdm import tqdm
from IPython import embed
import argparse

import glob
import re
import pickle as pkl
from easydict import EasyDict
import pprint

import random
import numpy as np
import pandas as pd 

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split, SubsetRandomSampler

from typing import Union, List, Optional

import ipywidgets as widgets
from ipywidgets import interact
import logging

from CardiacMotion.data.DataModules import CardiacMeshPopulationDM, CardiacMeshPopulationDataset
from CardiacMotion.utils.image_helpers import generate_gif, merge_gifs_horizontally
from CardiacMotion.utils.run_helpers import Run, get_model, get_runs, compute_thickness_per_aha
from CardioMesh.CardiacMesh import Cardiac3DMesh, transform_mesh
from copy import deepcopy
import matplotlib.pyplot as plt

partitions = {
  "left_atrium" : ("LA", "MVP", "PV1", "PV2", "PV3", "PV4", "PV5"),
  "right_atrium" : ("RA", "TVP", "PV6", "PV7"),
  "left_ventricle" : ("LV", "AVP", "MVP"),
  "right_ventricle" : ("RV", "PVP", "TVP"),
  "biventricle" : ("LV", "AVP", "MVP", "RV", "PVP", "TVP"),
  "aorta" : ("aorta",)
}

import trimesh
import multiprocessing

def generate_z_grid(run, n_zc, n_zs, min=-3, max=3, step=1):
    
    '''
    '''
    
    from itertools import product
    
    z_df = run.get_z_df()
    z_mean = z_df.mean()
    z_std = z_df.std()

    z_vars = [z for z in range(n_zc, n_zc+n_zs)]
    z_values = range(min, max+1, step)
    
    all_keys = []
    all_z = []
    
    for z_var, z_value in product(z_vars, z_values):       
        z = z_mean + z_value * np.diag(z_std)[z_var]
        z = torch.Tensor(z)
        key = (f"z{str(z_var).zfill(3)}", z_value)
        all_keys.append(key)
        all_z.append(z)
        
    all_z = torch.stack(all_z)
    
    z = EasyDict({"mu": all_z, "log_var": None})
    return all_keys, z
    

def generate_synthetic_shapes(run, z): # z_var, value, resolution=50):
    
    '''
    '''

    if isinstance(z, torch.Tensor):
        z = EasyDict({"mu": z, "log_var": None})
    
    if torch.cuda.is_available():
        z.mu = z.mu.to("cuda:0")
        output_mesh = run.model.decoder(z)[1].cpu()
    else:
        output_mesh = run.model.decoder(z)[1]
    
    output_mesh = output_mesh.detach().numpy()
    
    return output_mesh


def worker_function(args):

    synthetic_shapes_df, voxelsize, start, end = args
    full_cycle = []

    for i in range(start, end):
        # Perform your computation here
        vox = trimesh.Trimesh(synthetic_shapes_df.shapes.iloc[i], run.faces).voxelized(voxelsize)        
        full_cycle.append(vox)
        
    indices = synthetic_shapes_df.index[start:end]
    
    return pd.DataFrame(full_cycle, index=indices)


def generate_voxelized(run, synthetic_shapes, voxelsize=1, use_multiprocessing=True, num_cores=50, total_iterations=50, filename=None):
    
    '''
        run: a Run object (custom class defined in utils/run_helpers.py
        synthetic shapes: a 4-rank tensor (batch, time, vertex, coordinate)
    '''
    
    N_VERTICES = run.mean_shape.shape[0]
    z_keys_with_timeframe = [(z[0], z[1], t) for z in z_keys for t in range(50)]
    
    synthetic_shapes_df = pd.DataFrame(z_keys_with_timeframe).set_index([0, 1, 2])
    synthetic_shapes_df["shapes"] = [x for x in synthetic_shapes.reshape(-1, N_VERTICES, 3)]
    
    if use_multiprocessing:
        num_cores = num_cores
        total_iterations = total_iterations
        
        chunk_size = total_iterations // num_cores
        pool = multiprocessing.Pool(processes=num_cores)
        
        # Split the loop into chunks and assign them to different processes
        # print([(i, i + chunk_size) for i in range(0, total_iterations, chunk_size)])
        args = [(synthetic_shapes_df, voxelsize, i, i + chunk_size,) for i in range(0, total_iterations, chunk_size)]
        voxelized_shapes = pool.map(worker_function, args)
        
        pool.close()
        pool.join()
    else:
        raise NotImplementedError(f"This function has only been implemented for multiprocessing.")

    # Combine results from different processes
    voxelized_shapes = pd.concat(voxelized_shapes)
        
    # voxelized_shapes = parallel_for_loop(num_cores, kk.shape[0])
    
    voxelized_shapes.rename_axis(index={0: "z_var", 1: "value", 2: "timeframe"}, inplace=True)
    voxelized_shapes.rename({0: "voxelization"}, axis=1, inplace=True)
    
    vox_filled_shape = [ deepcopy(vox_shape.voxelization).fill() for index, vox_shape in voxelized_shapes.iterrows() ]    
    voxelized_shapes["filled_voxelization"] = vox_filled_shape

    if filename is not None:
        pkl.dump(voxelized_shapes, open(filename, "wb"))
    
    return voxelized_shapes


if __name__ == "__main__":

    MESHES_PATH = "/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/"
    
    fhm_mesh = Cardiac3DMesh(
      filename=f"{MESHES_PATH}/1000511/models/FHM_res_0.1_time001.npy",
      faces_filename=f"{os.environ['HOME']}/01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv",
      subpart_id_filename=f"{os.environ['HOME']}/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt"
    )
    
    runs_df = get_runs(experiment_ids=['4', '5'], from_cached=True, only_finished=True)
    
    previous_run, previous_expid = None, None
    
    voxelizations = {}
    
    for index, run_data in tqdm(runs_df.sort_values("experiment_id").iterrows()):    
                    
        run_id = index
        exp_id = run_data.experiment_id
        # chamber = run_data.partition
        
        vox_filename = f"/home/user/01_repos/CardiacMotion/data/cached/{run_id}_voxelizations.pkl"
        
        # if os.path.exists(vox_filename):
        #     logging.info(f"Run {exp_id}/{run_id} already has its voxelizations. Skipping...")
        #    continue
        
        # if exp_id != previous_expid:
        #     run.load_dataloader()
        #     dataloader = run.dataloader
        # else:
        #     run.dataloader = dataloader
    
        previous_expid, previous_run = exp_id, run_id
        
        n_zc, n_zs = int(run_data["params.latent_dim_c"]), int(run_data["params.latent_dim_s"])
        
        if n_zc == 8 and n_zs == 8:
            logging.info(f"Already done. Skipping...")
            continue

        run = Run(run_data, load_model=True, load_dataloader=False)
        
        try:
            z_keys, z_values = generate_z_grid(run, n_zc=n_zc, n_zs=n_zs, min=-3, max=3, step=3)
        except FileNotFoundError as e:
            logging.error(f"Error occurred: {e}")
            continue
        synthetic_shapes = generate_synthetic_shapes(run, z_values)    
        
        logging.info(f"Generating voxelizations...")
    
        voxelized_shapes = generate_voxelized(run, synthetic_shapes, voxelsize=1, filename=vox_filename,  total_iterations=len(z_keys) * 50)
        voxelizations[index] = voxelized_shapes
        
        logging.info(f"Finished voxelizations and saving to {vox_filename}")
        
        pkl.dump(voxelizations, open(vox_filename, "wb"))
