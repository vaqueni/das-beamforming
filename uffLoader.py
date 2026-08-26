import h5py
import os

# 현재 스크립트(uffLoader.py) 위치 기준
base_dir = os.path.dirname(os.path.abspath(__file__))  # .../PICMUS
uff_path = os.path.join(base_dir, "data", "PICMUS", "PICMUS_simulation_contrast_speckle.uff")
uff_path = os.path.normpath(uff_path)

f = h5py.File(uff_path, "r")

print(list(f.keys()))
print(list(f["channel_data"].keys()))

rf_data = f["channel_data"]["data"][()]
print(rf_data.shape)

c = f["channel_data"]["sound_speed"][()]
print(c)