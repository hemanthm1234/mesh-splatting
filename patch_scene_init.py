with open("/data1/hemanth/mesh-splatting/scene/__init__.py", "r") as f:
    code = f.read()

import re
old_logic = """        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
"""

new_logic = """        if vggt_args is not None and vggt_args.vggt_mode == "full_pipeline":
            print("Running in VGGT Omega Full Pipeline mode!")
            scene_info = sceneLoadTypeCallbacks["VGGT"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
"""

code = code.replace(old_logic, new_logic)

with open("/data1/hemanth/mesh-splatting/scene/__init__.py", "w") as f:
    f.write(code)
