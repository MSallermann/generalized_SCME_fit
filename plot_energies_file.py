from fit_scme import make_plots
import pandas as pd
from pathlib import Path


input_folder = Path("./saved_results/dimer_cluster_pbe_good_dimer_extended_core")
output_folder = input_folder
output_folder.mkdir(exist_ok=True)


df = pd.read_csv(input_folder / "energies_scme.csv")
make_plots(output_folder=output_folder, energies_scme_df=df, plot_initial=False)

print(f"Saved plot to {output_folder}/plot_energy.png")
print(f"Saved plot to {output_folder}/plot_residuals.png")

