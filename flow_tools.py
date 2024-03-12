import os
import warnings
import math
import inspect
from itertools import product
from socket import gethostname

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib as mpl
import matplotlib.pyplot as plt
import fcsparser as fcs
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

sns.set(style="white")
sns.set_context("notebook")
warnings.filterwarnings("once")
warnings.filterwarnings("ignore", category=DeprecationWarning)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

tick_setup = {'labelsize' : SMALL_SIZE,
              'major.pad': 0}

blue = '#0077bb'
cyan = '#33bbee'
teal = '#009988'
orange = '#ee7733'
red = '#cc3311'
magenta = '#ee3377'
grey = '#bbbbbb'
white = '#ffffff'
light_yellow = "#eecc66"
light_red = "#ee99aa"
light_blue = "#6699cc"
dark_yellow = "#997700"
dark_red = "#994455"
dark_blue = "#004488"


plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', **tick_setup)    # fontsize of the tick labels
plt.rc('ytick', **tick_setup)
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

outputdir = '/Users/weinberz/Library/CloudStorage/GoogleDrive-weinberz@gmail.com/.shortcut-targets-by-id/1yK71kpoZFkDwveJ6eqtJ9_7UygmXd3dJ/SynDevKit/Figures/jupyter out/'

def move_legend_fig_to_ax(fig, ax, loc, bbox_to_anchor=None, **kwargs):
    if fig.legends:
        fig.legends[0].set(visible=False)
        old_legend = fig.legends[-1]
    else:
        raise ValueError("Figure has no legend attached.")

    old_boxes = old_legend.get_children()[0].get_children()

    legend_kws = inspect.signature(mpl.legend.Legend).parameters
    props = {
        k: v for k, v in old_legend.properties().items() if k in legend_kws
    }

    props.pop("bbox_to_anchor")
    title = props.pop("title")
    if "title" in kwargs:
        title.set_text(kwargs.pop("title"))
    title_kwargs = {k: v for k, v in kwargs.items() if k.startswith("title_")}
    for key, val in title_kwargs.items():
        title.set(**{key[6:]: val})
        kwargs.pop(key)
    kwargs.setdefault("frameon", old_legend.legendPatch.get_visible())

    # Remove the old legend and create the new one
    props.update(kwargs)
    fig.legends = []
    new_legend = ax.legend(
        [], [], loc=loc, bbox_to_anchor=bbox_to_anchor, **props
    )
    new_legend.get_children()[0].get_children().extend(old_boxes)

def gate_density (df, threshold, y_var='FSC-A', x_var='SSC-A'):
    
    ymin = df[y_var].min()
    ymax = df[y_var].max()
    xmin = df[x_var].min()
    xmax = df[x_var].max()
                                                                                                                                
    #Perform a kernel density estimate on the data:
    df_mini = df.sample(1000)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([df_mini[x_var], df_mini[y_var]])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    cut = Z.max()*threshold
    
    # threhold dataframe
    df['gate'] = kernel([df[x_var], df[y_var]])
    df2 = df[df.gate > cut]
    return df2

def load_data(folder, colnames):
    ffolderlist = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    df=pd.DataFrame()
    aa=1
        #load all fcs files
    for fdn in ffolderlist:
        ffilelist = os.listdir(folder+fdn)
        for fn in ffilelist:
            filename, file_ext = os.path.splitext(fn)
            if file_ext ==".fcs":
                path = folder + fdn + "//" + fn
                meta, df1 = fcs.parse(path, meta_data_only=False, reformat_meta=True)
                df1["WellName"]=fn.split("_")[3].split(".")[0]
                df1["WellNumber"]= aa
                df=df.append(df1)
                aa=aa+1

    # metadata file
    meta = folder+ "metadata.csv"
    df1=pd.read_csv(meta, names=colnames)

    df=df1.merge(df, on=["WellName"])
    
    return df

def clean_data(df):
    #remove small events
    df2 =df[df["FSC-A"]>5E4]
    df2 =df2[df2["SSC-A"]>2.5E4]

    #remove NAs
    df2= df2.loc[(df2.select_dtypes(include=['number']) > 0).all(axis='columns'), :]
    df2=df2.dropna()

    #gate cells
    df2 = gate_density(df2, 0.1)

    #log data
    for col in df2.columns:
        if df2[col].dtype=="float32":
            df2["log"+ col]=df2[col].apply(math.log10)
    
    return df2