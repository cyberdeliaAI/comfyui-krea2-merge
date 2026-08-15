"""
@author: Cyberdelia
@title: ComfyUI Krea2 Merge
@nickname: Krea2 Merge
@description: Load, merge, and save Krea 2 PEFT/Diffusers and Kohya LoRAs.
"""

import importlib


MODULES = ["mergetools.merge_lora_tools"]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_path in MODULES:
    try:
        imported_module = importlib.import_module(f".{module_path}", __name__)
        NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
    except ImportError as error:
        print(f"Krea2 Merge: unable to import {module_path}: {error}")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
