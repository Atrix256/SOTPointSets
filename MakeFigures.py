import sys
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

fileNames = glob.glob("out/*.png")

for fileName in fileNames:
    print(fileName)

    loadedImage = Image.open(fileName)
    
    im = np.array(loadedImage, dtype=float) / 255.0
    
    # get the DFT magnitude, zero out DC and shift it so DC is in the middle
    dft = abs(np.fft.fft2(im))
    dft[0,0] = 0.0
    dft = np.fft.fftshift(dft)

    # log and normalize DFT
    imOut = np.log(1+dft)
    themin = np.min(imOut)
    themax = np.max(imOut)
    if themin != themax:
        imOut = (imOut - themin) / (themax - themin)
    else:
        imOut = imOut

    # Write out DFT
    outFileName = fileName + ".magnitude.png"
    Image.fromarray((imOut*255.0).astype(np.uint8), mode="L").save(outFileName)
    
    # Tile image as 3x3
    imOut = Image.new(loadedImage.mode, (loadedImage.size[0] * 3, loadedImage.size[1] * 3), 255)
    for i in range(3):
        for j in range(3):
            imOut.paste(loadedImage, (i*loadedImage.size[0],j*loadedImage.size[1]))
    imOut.save(fileName + ".3x3.png")
    
    # Tile image as 11x11
    imOut = Image.new(loadedImage.mode, (loadedImage.size[0] * 11, loadedImage.size[1] * 11), 255)
    for i in range(11):
        for j in range(11):
            imOut.paste(loadedImage, (i*loadedImage.size[0],j*loadedImage.size[1]))
    imOut.save(fileName + ".11x11.png")


fileNames = glob.glob("out/*.csv")

for fileName in fileNames:
    print(fileName)

    fig, ax = plt.subplots()
    df = pd.read_csv(fileName).drop(['Iteration'], axis=1)

    ax.plot(df['Avg. Movement'], label="Avg. Movement")

    plt.title('Log/Log Average Movement Each Iteration: ' + fileName)

    fig.axes[0].set_xscale('log', base=2)
    fig.axes[0].set_yscale('log', base=2)

    fig.tight_layout()
    fig.savefig(fileName + ".graph.png", bbox_inches='tight')
