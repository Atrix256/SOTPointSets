import sys
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

fileNames =[
    "out/batch1_square.csv",
    "out/batch4_square.csv",
    "out/batch16_square.csv",
    "out/batch64_square.csv",
    "out/batch256_square.csv",
]

fig, ax = plt.subplots()
plt.title('Log/Log Average Movement Each Iteration')

xValues = range(1, 64000, 634)

for fileName in fileNames:
    print(fileName)
    df = pd.read_csv(fileName).drop(['Iteration'], axis=1)
    ax.plot(xValues, df['Avg. Movement'], label=fileName)

ax.legend()

fig.axes[0].set_xscale('log', base=2)
fig.axes[0].set_yscale('log', base=2)

fig.tight_layout()
fig.savefig("out/batch_square.graph.png", bbox_inches='tight')



fileNames =[
    "out/batch1_stratified_square.csv",
    "out/batch4_stratified_square.csv",
    "out/batch16_stratified_square.csv",
    "out/batch64_stratified_square.csv",
    "out/batch256_stratified_square.csv",
]

fig, ax = plt.subplots()
plt.title('Log/Log Average Movement Each Iteration')

xValues = range(1, 64000, 634)

for fileName in fileNames:
    print(fileName)
    df = pd.read_csv(fileName).drop(['Iteration'], axis=1)
    ax.plot(xValues, df['Avg. Movement'], label=fileName)

ax.legend()

fig.axes[0].set_xscale('log', base=2)
fig.axes[0].set_yscale('log', base=2)

fig.tight_layout()
fig.savefig("out/batch_stratified_square.graph.png", bbox_inches='tight')



fileNames =[
    "out/square.csv",
    "out/circle.csv",
    "out/square_GR.csv",
    "out/circle_GR.csv",
]

fig, ax = plt.subplots()
plt.title('Log/Log Average Movement Each Iteration')

xValues = range(1, 64000, 634)

for fileName in fileNames:
    print(fileName)
    df = pd.read_csv(fileName).drop(['Iteration'], axis=1)
    ax.plot(xValues, df['Avg. Movement'], label=fileName)

ax.legend()

fig.axes[0].set_xscale('log', base=2)
fig.axes[0].set_yscale('log', base=2)

fig.tight_layout()
fig.savefig("out/GR.graph.png", bbox_inches='tight')
