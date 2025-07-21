import os
import subprocess

strategies = {
    "fixedpoint_only": "fixedpointonly",
    "floatingpoint_only": "floatingpointonly",
    "fixed_floating": "fixedfloating"
}

for strategy, subdir in strategies.items():
    print(f"Processing strategy: {strategy}")

    if not os.path.isdir(subdir):
        print(f"Directory '{subdir}' does not exist. Skipping.")
        continue
    os.chdir(subdir)

    try:
        subprocess.run(f"taffo -O3 -DM=10000 -fno-vectorize -fno-slp-vectorize -I/home/riccardo/Desktop/TAFFO/TAFFO_dta_fork/test/fpbench/turbine3 -time-profile-file /home/riccardo/Desktop/TAFFO/TAFFO_dta_fork/test/fpbench/turbine3/turbine3_taffo_time.csv   /home/riccardo/Desktop/TAFFO/TAFFO_dta_fork/test/fpbench/turbine3/turbine3.c -o turbine3-taffo -lm -Xdta --dtastrategy={strategy}", check=True, shell=True)
        print(f"Command for strategy '{strategy}' completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Command failed for strategy '{strategy}': {e}")

    os.chdir("..")
