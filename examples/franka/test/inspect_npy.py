import numpy as np
from typing import Any

def analyze_structure(obj: Any, indent: int = 0) -> str:
    """
    Recursively analyzes the data structure and generates a description similar to a Dataset format.
    
    Args:
        obj: The object to analyze.
        indent: The indentation level.
    
    Returns:
        str: The formatted structure description.
    """
    spaces = "    " * indent
    
    if isinstance(obj, dict):
        result = "{\n"
        for key, value in obj.items():
            result += f"{spaces}    '{key}': "
            
            # If the value is a dictionary, process it recursively
            if isinstance(value, dict):
                result += analyze_structure(value, indent + 1)
            else:
                result += analyze_single_item(value)
            result += ",\n"
        result += f"{spaces}}}"
        return result
    
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            # Analyze the structure of the first element in the list
            first_item = obj[0]
            return f"List[{analyze_single_item(first_item)}] (length={len(obj)})"
        else:
            return "List[] (empty)"
    
    else:
        return analyze_single_item(obj)

def analyze_single_item(obj: Any) -> str:
    """
    Analyzes the type and shape of a single data item.
    
    Args:
        obj: The object to analyze.
    
    Returns:
        str: The formatted type description.
    """
    if isinstance(obj, np.ndarray):
        dtype_str = str(obj.dtype)
        
        # Determine the data type
        if dtype_str.startswith('float'):
            type_name = "Tensor"
        elif dtype_str.startswith('int'):
            type_name = "Tensor" 
        elif dtype_str.startswith('uint8') and len(obj.shape) == 3:
            type_name = "Image"
        elif dtype_str.startswith('bool'):
            type_name = "Tensor"
        elif obj.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, object
            type_name = "Text"
        else:
            type_name = "Tensor"
        
        return f"{type_name}(shape={obj.shape}, dtype={dtype_str})"
    
    elif isinstance(obj, str):
        return f"Text(shape=(), dtype=string)"
    
    elif isinstance(obj, (int, float)):
        return f"Scalar(dtype={type(obj).__name__})"
    
    elif isinstance(obj, bool):
        return f"Scalar(dtype=bool)"
    
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            return f"List[{analyze_single_item(obj[0])}] (length={len(obj)})"
        else:
            return "List[] (empty)"
    
    elif isinstance(obj, dict):
        return "FeaturesDict({\n" + analyze_structure(obj, 1) + "\n})"
    
    else:
        return f"Unknown(type={type(obj).__name__})"

def analyze_npy_structure(file_path: str) -> None:
    """
    Analyzes the data structure of a .npy file.
    
    Args:
        file_path: The path to the .npy file.
    """
    try:
        # Load the data
        data = np.load(file_path, allow_pickle=True).item()
        print(f"File: {file_path}")
        print(f"Top-level type: {type(data)}")
        print("\nData Structure:")
        print("=" * 50)
        
        if isinstance(data, dict):
            structure = analyze_structure(data)
            print(f"Dataset({structure})")
        else:
            print(f"Top-level is not a dictionary, but: {type(data)}")
            print(f"Structure: {analyze_single_item(data)}")
            
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    file_path = "/nvme_data/bingwen/share_datasets/franka_panda/pick_to_plate-real/episode_1/data.npy"
    analyze_npy_structure(file_path)
    
    # # for specific key analysis, uncomment the following lines
    # try:
    #     data = np.load(file_path, allow_pickle=True).item()
    #     print("\n\n详细分析各个键:")
    #     print("=" * 50)
    #     for key, value in data.items():
    #         print(f"\n'{key}':")
    #         print(f"  类型: {type(value)}")
    #         if hasattr(value, 'shape'):
    #             print(f"  形状: {value.shape}")
    #         if hasattr(value, 'dtype'):
    #             print(f"  数据类型: {value.dtype}")
    #         print(f"  结构: {analyze_single_item(value)}")
    # except Exception as e:
    #     print(f"详细分析时出错: {e}")