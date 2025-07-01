"""
使用 tfds.load 将一个标准的 RLDS 数据集转换为 LeRobot 格式的脚本。

此脚本适用于已通过 `tensorflow_datasets` 工具正确生成元数据的数据集。

用法:
python your_script_name.py --dataset-name "panda_rlds_dataset" --data-dir "/path/to/your/parent/directory"

示例:
python convert_my_data.py --dataset-name "panda_rlds_dataset" --data-dir "/nvme_data/bingwen/share_datasets/franka_panda/franka_panda_pick_to_plate-real"

您也可以将最终的数据集推送到 Hugging Face Hub:
python convert_my_data.py --dataset-name "panda_rlds_dataset" --data-dir "/path/to/your/data" --push-to-hub

注意: 运行此脚本前，请确保已安装以下包:
`pip install lerobot tensorflow tensorflow-datasets tyro`
"""
import shutil
import os
import tyro
import tensorflow as tf
import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


def main(repo_id: str, dataset_name: str, data_dir: str, *, push_to_hub: bool = False):
    """
    主转换函数。

    Args:
        dataset_name: 要加载的 TFDS 数据集名称 (例如 "panda_rlds_dataset")。
        data_dir: 包含 TFDS 数据集文件夹的父目录。
        push_to_hub: 如果为 True, 则将转换后的数据集推送到 Hugging Face Hub。
    """

    output_path = HF_LEROBOT_HOME / repo_id # HF_LEROBOT_HOME is defined for LeRobotDataset
    if output_path.exists():
        print(f"正在移除已存在的数据集: {output_path}")
        shutil.rmtree(output_path)


    print("正在创建 LeRobot 数据集结构...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="franka_panda",
        fps=5,  # todo?
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # --- 3. 使用 tfds.load 加载原始数据集 ---
    print(f"正在从目录 '{data_dir}' 加载 TFDS 数据集: {dataset_name}")
    try:
        raw_dataset, ds_info= tfds.load(name=dataset_name, data_dir=data_dir, split="train", with_info=True)
        print("\n=== Dataset Info ===")
        print(ds_info)
    except Exception as e:
        print(f"错误: 使用 tfds.load 加载数据集失败。请检查 `dataset_name` 和 `data_dir` 是否正确。")
        print(f"TensorFlow 错误信息: {e}")
        return

    # --- 4. 遍历数据并写入 LeRobot 数据集 ---
    print("开始转换流程...")
    for episode in raw_dataset:
        # RLDS 格式中，每个 episode 包含一个 'steps' 数据集
        # 我们需要遍历这些 step 来构建 LeRobot 的 episode
        num_steps = 0
        language_instruction = None
        for step in episode["steps"].as_numpy_iterator():
            if language_instruction is None:
                language_instruction = step["language_instruction"].decode('utf-8')
            dataset.add_frame(
                {
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                    "task": language_instruction,
                }
            )
            num_steps += 1

        if num_steps > 0: # only the episode is valid.
            dataset.save_episode()
            print(f"已保存任务 '{language_instruction}' 的 episode (包含 {num_steps} 个步骤)")
        else:
            print("发现一个空 episode, 已跳过。")

    if push_to_hub:
        print("正在推送到 Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["panda", "rlds", "robotics"], 
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    print("\n转换完成!")
    print(f"LeRobot 数据集已保存至: {output_path}")

if __name__ == "__main__":
    # 使用 tyro 从 main 函数的参数自动创建命令行接口
    tyro.cli(main)