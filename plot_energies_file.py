from fit_scme import make_plots
import pandas as pd
from pathlib import Path

path_to_csv = Path(
    "./saved_results/dimer_cluster_pbe_good_dimer_extended_core/energies_scme.csv"
)
output_folder = Path("./temp_plot")
output_folder.mkdir(exist_ok=True)


df = pd.read_csv(path_to_csv)
make_plots(output_folder=output_folder, energies_scme_df=df, plot_initial=False)

print(f"Saved plot to {output_folder}/plot_energy.png")