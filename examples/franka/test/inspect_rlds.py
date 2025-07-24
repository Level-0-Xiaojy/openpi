import tensorflow_datasets as tfds
import tensorflow as tf

# 设置数据路径和数据集名
DATASET_NAME = "panda_rlds_dataset"
DATA_DIR = "/nvme_data/bingwen/share_datasets/franka_panda/franka_panda_pick_to_plate-real"
SPLIT = "train"

# 加载数据集及其信息
print(f"Loading TFDS dataset: {DATASET_NAME}")
ds, ds_info = tfds.load(
    name=DATASET_NAME,
    split=SPLIT,
    data_dir=DATA_DIR,
    with_info=True
)

print("\n=== Dataset Info ===")
print(ds_info)


# === 定义 print_nested_dict 函数 ===
def print_nested_dict(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            print_nested_dict(v, prefix + k + "/")
    elif isinstance(d, tf.data.Dataset):
        print(f"{prefix[:-1]:<40} | tf.data.Dataset (sequence of steps)")
        for step in d.take(1):
            print("  └─ One step in steps:")
            for sk, sv in step.items():
                if isinstance(sv, dict):
                    for subk, subv in sv.items():
                        print(f"     steps/{sk}/{subk:<16} | shape={subv.shape}, dtype={subv.dtype}")
                else:
                    print(f"     steps/{sk:<20} | shape={sv.shape}, dtype={sv.dtype}")
        print("  ...")
    else:
        print(f"{prefix[:-1]:<40} | shape={d.shape}, dtype={d.dtype}")


for example in ds.take(1):
    print("\n=== Sample Structure ===")
    print_nested_dict(example)

