"""Voxelize + batch a list of SemanticKITTI scans.

Voxelization is pure-numpy (floor + np.unique) so it does NOT depend on a specific torchsparse
version — only the backbone touches torchsparse. Predictions are made at voxel level and mapped
back to points via `inverse` for point-level loss / eval.

Returned batch:
  coords   [Vtot, 4] int   (x, y, z, batch)   # torchsparse SparseTensor coords — verify col order for your version
  feats    [Vtot, Cin] f32 (representative point feature per voxel)
  inverse  [Ptot] long      each point -> its global voxel row
  sem      [Ptot] long      0..19
  xyz      [Ptot, 3] f32
  inst     [Ptot] long
  pbatch   [Ptot] long       sample index per point (for per-scan clustering/eval)
  meta     list[(seq, frame, n_points)]
"""

from __future__ import annotations

import numpy as np
import torch


def voxelize_collate(samples: list[dict], voxel: float, in_channels: int = 4) -> dict:
    coords_all, feats_all = [], []
    inverse_all, sem_all, xyz_all, inst_all, pbatch_all = [], [], [], [], []
    meta = []
    vox_offset = 0

    for b, s in enumerate(samples):
        xyz = s["xyz"].astype(np.float32)  # [Np,3]
        # point feature: [x,y,z,remission] (Cin=4) or [remission] (Cin=1)
        pfeat = (
            np.concatenate([xyz, s["feat"].astype(np.float32)], axis=1)
            if in_channels == 4
            else s["feat"].astype(np.float32)
        )
        vc = np.floor(xyz / voxel).astype(np.int32)  # [Np,3]
        # torchsparse expects non-negative voxel coords; KITTI has points behind/left/below the
        # sensor (negative xyz). Shift each scan to its own min so the sparse-conv kernel maps stay
        # valid at deep downsampling levels. Geometry is unchanged (clustering uses float xyz).
        vc -= vc.min(axis=0, keepdims=True)
        uniq, first_idx, inverse = np.unique(vc, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)

        nv = uniq.shape[0]
        bcol = np.full((nv, 1), b, dtype=np.int32)
        coords_all.append(np.concatenate([uniq, bcol], axis=1))  # [nv,4]
        feats_all.append(pfeat[first_idx])                       # representative feat per voxel

        inverse_all.append(inverse + vox_offset)                 # -> global voxel rows
        vox_offset += nv

        sem_all.append(s.get("sem", np.zeros(len(xyz), np.int64)))
        inst_all.append(s.get("inst", np.zeros(len(xyz), np.int64)))
        xyz_all.append(xyz)
        pbatch_all.append(np.full(len(xyz), b, dtype=np.int64))
        meta.append((s["seq"], s["frame"], len(xyz)))

    return {
        "coords": torch.from_numpy(np.concatenate(coords_all)).int(),
        "feats": torch.from_numpy(np.concatenate(feats_all)).float(),
        "inverse": torch.from_numpy(np.concatenate(inverse_all)).long(),
        "sem": torch.from_numpy(np.concatenate(sem_all)).long(),
        "inst": torch.from_numpy(np.concatenate(inst_all)).long(),
        "xyz": torch.from_numpy(np.concatenate(xyz_all)).float(),
        "pbatch": torch.from_numpy(np.concatenate(pbatch_all)).long(),
        "meta": meta,
    }
