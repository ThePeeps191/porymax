import os
import stat
import inspect
from huggingface_hub import hf_hub_download
import amago.nets.transformer as amago_transformer

def patch_kakuna_config():
    print("Locating Kakuna's config.txt in HuggingFace cache...")
    
    # Find the exact file path using the same library Metamon uses
    config_path = hf_hub_download(
        repo_id="jakegrigsby/metamon",
        filename="kakuna/ckpts/config.txt",
        repo_type="model"
    )
    print(f"Found config.txt at:\n{config_path}\n")
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "FlashAttention" not in content:
        print("FlashAttention not found in config. It is already patched.")
        return

    # Find the correct CPU-friendly attention class in Amago
    classes = [name for name, obj in inspect.getmembers(amago_transformer) if inspect.isclass(obj)]
    print(f"Available attention classes in Amago: {classes}")
    fallback_class = "MultiheadAttention" if "MultiheadAttention" in classes else "VanillaAttention"
    print(f"Replacing '@transformer.FlashAttention' with '@transformer.{fallback_class}'...")
    new_content = content.replace("@transformer.FlashAttention", f"@transformer.{fallback_class}")
    
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except PermissionError:
        print("Cache file is read-only. Fixing permissions...")
        os.chmod(config_path, stat.S_IWRITE)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
    print("Successfully patched the cached config.txt.")

if __name__ == "__main__":
    patch_kakuna_config()