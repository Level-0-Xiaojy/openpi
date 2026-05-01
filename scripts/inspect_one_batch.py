import argparse
import dataclasses
import json
import os

from openpi.training import config as cfg
from openpi.training import data_loader as dl


def _item(x):
    try:
        return x.item()
    except Exception:
        return x


def _to_jsonable(x, *, max_list: int = 50):
    """Convert common array/scalar containers into JSON-friendly structures."""
    # Basic python scalars
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # bytes-like
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return str(x)

    # numpy/jax scalars
    try:
        item = x.item()  # type: ignore[attr-defined]
        if isinstance(item, (bool, int, float, str)):
            return item
    except Exception:
        pass

    # dict/list/tuple
    if isinstance(x, dict):
        return {k: _to_jsonable(v, max_list=max_list) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v, max_list=max_list) for v in x]

    # array-like (numpy/jax/torch) -> store shape/dtype/head
    shape = getattr(x, "shape", None)
    dtype = getattr(x, "dtype", None)
    if shape is not None:
        try:
            import numpy as np

            arr = np.asarray(x)
            flat = arr.reshape(-1)
            head = flat[:max_list].tolist()
            return {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype) if dtype is not None else str(type(x)),
                "head": head,
            }
        except Exception:
            return {
                "shape": list(shape) if isinstance(shape, (list, tuple)) else str(shape),
                "dtype": str(dtype) if dtype is not None else str(type(x)),
            }

    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="训练 config 名，例如 test_sm2sm")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--num-items", type=int, default=5)
    ap.add_argument(
        "--out",
        default=None,
        help="输出 JSON 路径（默认写到 debug_dumps/inspect_<config>_<split>.json）",
    )

    # 是否开启 prompt_from_task（默认不改，保持 config 原样）
    ap.add_argument("--enable-prompt-from-task", action="store_true")
    # meta 拼接/dropout（默认不改，保持 config 原样）
    ap.add_argument("--meta-key", default=None, help="例如 meta；不传则保持 config 原样")
    ap.add_argument("--meta-dropout-p", type=float, default=None, help="不传则保持 config 原样")
    ap.add_argument("--meta-dropout-seed", type=int, default=None, help="不传则保持 config 原样")

    args = ap.parse_args()

    train_cfg = cfg.get_config(args.config)

    # 构建 DataConfig（frozen，不可直接赋值，必须 dataclasses.replace）
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    replace_kwargs = {}

    if args.enable_prompt_from_task:
        replace_kwargs["prompt_from_task"] = True
    if args.meta_key is not None:
        replace_kwargs["prompt_meta_key"] = args.meta_key
    if args.meta_dropout_p is not None:
        replace_kwargs["prompt_meta_dropout_p"] = float(args.meta_dropout_p)
    if args.meta_dropout_seed is not None:
        replace_kwargs["prompt_meta_dropout_seed"] = int(args.meta_dropout_seed)

    if replace_kwargs:
        data_cfg = dataclasses.replace(data_cfg, **replace_kwargs)

    # ===== [A] 看 prompt/meta 拼接(dropout)后的“样本级数据”（更直观）=====
    print("=== [A] dataset items (after prompt/meta transforms) ===")
    ds = dl.create_torch_dataset(
        data_cfg,
        train_cfg.model.action_horizon,
        train_cfg.model,
        split=args.split,
    )

    n = min(args.num_items, len(ds))
    dump = {
        "config": args.config,
        "split": args.split,
        "A_items": [],
        "B_batch": {},
    }
    for i in range(n):
        item = ds[i]
        keys = sorted(item.keys())
        prompt = item.get("prompt", None)
        meta = item.get(data_cfg.prompt_meta_key, None) if getattr(data_cfg, "prompt_meta_key", None) else item.get("meta", None)

        print(f"\n[{i}] keys={keys}")
        for k in ["episode_index", "frame_index", "task_index"]:
            if k in item:
                print(f"  {k}:", _item(item[k]))
        if meta is not None:
            print("  meta :", _item(meta))
        else:
            print("  meta : <missing>")
        if prompt is not None:
            print("  prompt:", _item(prompt))
        else:
            print("  prompt: <missing>")

        dump["A_items"].append(_to_jsonable(item))

    # ===== [B] 看“最终送入训练”的 batch（全 pipeline）=====
    print("\n=== [B] one batch (after full pipeline) ===")
    # 注意：这里用 train_cfg（原始 config）建 loader，不会自动带上你在上面 dataclasses.replace 的 data_cfg 覆盖。
    # 所以 [B] 用来回答“真实训练现在到底喂了什么”，[A] 用来让你调参观察 prompt/meta 行为。
    loader = dl.create_data_loader(
        train_cfg,
        framework="jax",
        split=args.split,
        shuffle=False,
        num_batches=1,
        skip_norm_stats=True,
    )
    obs, actions = next(iter(loader))
    d = obs.to_dict()

    print("observation keys:", sorted(d.keys()))
    print("has prompt:", "prompt" in d)
    print("has tokenized_prompt:", "tokenized_prompt" in d)
    if "prompt" in d:
        try:
            print("prompt value:", d["prompt"] if isinstance(d["prompt"], str) else d["prompt"].item())
        except Exception as e:
            print("prompt value read failed:", e)
    print("actions shape:", getattr(actions, "shape", None))

    dump["B_batch"]["observation"] = _to_jsonable(d)
    dump["B_batch"]["actions"] = _to_jsonable(actions)

    out_path = args.out or f"debug_dumps/inspect_{args.config}_{args.split}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    print(f"\nWrote JSON to: {out_path}")


if __name__ == "__main__":
    main()