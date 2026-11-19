# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:19:31 2026

@author: tizia
"""

import pickle
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib
"""
### Save figures with segmentation result
"""
def format_scene(name):
    import re
    match = re.match(r"(Scene)0*(\d+)", name)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}"
    return name

font = {'family': 'sans-serif', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)
# load segmentation results
type_cell = "HPC_3000"
n_tile = "075"
n_well = "c3"
path= rf"C:\Users\tizia\OneDrive - INSTITUT CURIE\UnitPC_training\Results\MB03102025_new\VisioWell - {type_cell}_0degrees"
file = f"cellinfoMB_250807_sherlock_MB_251003_cord_blood_34-01({type_cell})_tile{n_tile}_{n_well}.pkl"
with open(os.path.join(path, file), mode='rb') as f:
    data = pickle.load(f)
    
# load well result
path_well = rf"C:\Users\tizia\OneDrive - INSTITUT CURIE\UnitPC_training\Wells\MB03102025_new\{type_cell}_0degrees"
name_well = f"sherlock_MB_251003_cord_blood_34-01({type_cell})_tile{n_tile}.pkl"
with open(os.path.join(path_well, name_well), mode='rb') as d:
    data_well = pickle.load(d)
    
dw = data_well[n_well]["crop"]
masks = data["masks"]

idx_to_save = [0, 52, 70, 100, 120]

for i in range(len(masks)):
    img = dw[:, :, i]
    mask = masks[i]

    H, W = img.shape

    mask_resized = cv2.resize(
        mask,
        (W, H),              
        interpolation=cv2.INTER_NEAREST
    )
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.imshow(mask_resized, cmap='Reds', alpha=0.4)
    plt.title(f"Frame {i+1}", fontsize=40)
    plt.axis('off')
    
    if i in idx_to_save:
        filename = f'mask_output_{i:03d}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Image saved: {filename} (300 DPI)")
        
    plt.show()


"""
### Save figures with curve
"""
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker 

path= rf"C:\Users\tizia\OneDrive - INSTITUT CURIE\UnitPC_training\Results\MB03102025_new\VisioWell - {type_cell}_0degrees"
file_csv = f"output_MB_250807_sherlock_MB_251003_cord_blood_34-01({type_cell})_tile{n_tile}_{n_well}.csv"

data_csv = pd.read_csv(os.path.join(path, file_csv), sep=";")

# variables to plot
y1 = data_csv["Output YOLO"]
y2 = data_csv["Correction"]
assert len(y1)==len(y2)
x = np.arange(1, len(y1)+1)



font = {'family': 'sans-serif', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(x, y1, label='Output Segmentation', color='#BBBBBC', linewidth=4, linestyle='-')
ax.plot(x, y2, label='Correction', color="#5AAA46", linewidth=2, linestyle='-')

ax.set_xlabel('Frame', fontweight='bold')
ax.set_ylabel('Cell Number', fontweight='bold')
ax.set_title('Experiment result', pad=15)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.grid(True, linestyle=':', alpha=0.6)

ax.set_xlim(x[0], x[-1])

ax.legend(loc='best', fontsize=14)
plt.tight_layout() 


plt.savefig('figure_panel_3d.png', dpi=300, bbox_inches='tight')

plt.show()
 