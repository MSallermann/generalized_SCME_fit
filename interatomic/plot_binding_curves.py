import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class PlotInfo:
    data: npt.ArrayLike
    name: Optional[str] = None


def plot(ax, plot_info: PlotInfo):
    distance = plot_info.data[:, 0]
    energy = plot_info.data[:, 1]
    ax.plot(distance, energy, label=plot_info.name)


data_pbe = np.loadtxt("./resources/PBE/PES_dimer_c1_PBE.txt")
plot_info_pbe = PlotInfo(data=data_pbe, name="PBE")

data_beef = np.loadtxt("./resources/BEEF-vdW/PES_dimer_c1_BEEF-vdW.txt")
plot_info_beef = PlotInfo(data=data_beef, name="Beef")

plot_info_list = [plot_info_pbe, plot_info_beef]

for p in Path(".").glob("binding_curve*/results.txt"):
    data = np.loadtxt(p)
    name = p.name
    plot_info_list.append(PlotInfo(data=data, name=name))

for plot_info in plot_info_list:
    plot(ax=plt.gca(), plot_info=plot_info)

plt.legend()
plt.show()
