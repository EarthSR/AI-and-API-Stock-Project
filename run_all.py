import subprocess
import os

base = r"C:\Users\EarthSR\AI and API Stock Project"

# Run LSTM
subprocess.run(["python", "LSTM_model.py"], cwd=os.path.join(base, "LSTM_model"))

# Run GRU
subprocess.run(["python", "GRU_model.py"], cwd=os.path.join(base, "GRU_Model"))
