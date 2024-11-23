import os, sys; sys.path.append(f"{os.environ['HOME']}/01_repos")
import argparse
from CardiacMotion.utils.run_helpers import Run, get_runs

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    args = parser.parse_args()

    runs_df = get_runs(only_finished=True)
    
    for i in range(len(runs_df)):
        run = Run(runs_df.iloc[i], batch_size=args.batch_size)
        run.generate_z_df()