from pathlib import Path
from blenderproc.python.types.MeshObjectUtility import MeshObject
import blenderproc as bproc
import random
import numpy as np
import argparse


def load_objs(path):
    objs = []
    path = Path(path)
    for f in path.iterdir():
        if f.is_file():
            # mesh_objs = bproc.loader.load_blend(str(f)) if f.suffix == '.blend' else
            ent_objs = bproc.loader.load_blend(str(f)) if f.suffix == '.blend' \
                else bproc.loader.load_obj(str(f))  # we load blend and obj
            mesh_objs = []
            for obj in ent_objs:
                if isinstance(obj, MeshObject):
                    mesh_objs.append(obj)
                else:
                    obj.deselect()
                    obj.delete()
            if len(mesh_objs) == 0:
                continue
            if len(mesh_objs) > 1:
                mesh_objs[0].join_with_other_objects(mesh_objs[1:])
                mesh_objs[0].select()
                # for obj in mesh_objs[1:]:
                #    obj.deselect()
                #    obj.delete()
            objs.append(mesh_objs[0])
    return objs


def convert_entity_to_mesh(entity):
    '''
    blender_objs = [e.blender_obj for e in entities]
    mesh_objs = bproc.object.convert_to_meshes(blender_objs)
    return mesh_objs
    '''
    blender_obj = entity.blender_obj
    mesh_obj = MeshObject(blender_obj)
    return mesh_obj


def sample_material(selected_objs):
    for obj in selected_objs:
        if len(obj.get_materials()) > 0:
            mat = obj.get_materials()[0]
            mat.set_principled_shader_value("Specular", random.uniform(0, 1))
            mat.set_principled_shader_value("Roughness", random.uniform(0, 1))
            mat.set_principled_shader_value("Base Color", np.random.uniform([0, 0, 0, 1], [1, 1, 1, 1]))
            mat.set_principled_shader_value("Metallic", random.uniform(0, 1))


def _get_random_range(flags, random_ranges, fix_points):
    l = len(flags)
    res = [random_ranges[i] if flags[i] else fix_points[i] for i in range(l)]
    return res


def generate_sample_pose_fn(config):
    location_cfg = config.location.random
    [range_min, range_max] = _get_xyz_range(location_cfg, [[-3, 3]] * 3, [[-8, -8], [0, 0], [0, 0]])

    def sample_pose(_obj: bproc.types.MeshObject):
        _obj.set_location(np.random.uniform(range_min, range_max))
        _obj.set_rotation_euler(
            bproc.sampler.uniformSO3(config.rotation.random.x, config.rotation.random.y, config.rotation.random.z))

    return sample_pose


def sample_light(_light):
    _light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    _light.set_location(np.random.uniform([-5, -5, 0], [5, 5, 12]))
    _light.set_energy(np.random.uniform(100, 1000))


def sample_camera(_poi, config):
    location_cfg = config.location.random
    [elevation, azimuth] = _get_random_range([location_cfg.elevation, location_cfg.azimuth], [[-45, 45], [-90, 90]],
                                             [[25, 25.1], [0, 0.01]])
    location = bproc.sampler.shell(center=[0, 0, 0],
                                   radius_min=3,
                                   radius_max=5,
                                   elevation_min=elevation[0],
                                   elevation_max=elevation[1],
                                   azimuth_min=azimuth[0],
                                   azimuth_max=azimuth[1])
    look_at_cfg = config.look_at.random
    range_min, range_max = _get_xyz_range(look_at_cfg, [[-1, 1]] * 3, [[0, 0]] * 3)
    lookat_point = _poi + np.random.uniform(range_min, range_max)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(lookat_point - location,
                                                             inplane_rot=np.random.uniform(-0.7854, 0.7854)
                                                             if config.in_plane_rotation.random else 0.0)
    # Add homog cam pose based on location a rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    return cam2world_matrix


def _get_xyz_range(cfg, random_ranges, fix_points):
    [range_x, range_y, range_z] = _get_random_range([cfg.x, cfg.y, cfg.z], random_ranges,
                                                    fix_points)
    range_min = [range_x[0], range_y[0], range_z[0]]
    range_max = [range_x[1], range_y[1], range_z[1]]
    return range_min, range_max


def random_material_texture_infusion(materials, selected_objs, texture):
    for mat in materials:
        if np.random.uniform(0, 1) <= 0.5:
            t = random.choice(texture)
            mat.infuse_texture(t, mode="set")
    for obj in selected_objs:
        for i in range(len(obj.get_materials())):
            # In 50% of all cases
            # if np.random.uniform(0, 1) <= 0.5:
            # Replace the material with a random one
            m = random.choice(materials)
            obj.set_material(i, m)


def check_material(base_material, materials):
    for mat in materials:
        try:
            mat.get_the_one_node_with_type("BsdfPrincipled")
        except Exception:
            mat.update_blender_ref(base_material)


def random_select_obj(objs, p_select, p_not_select):
    mask = np.random.choice(a=[False, True], size=(len(objs),), p=[p_not_select, p_select])
    selected_objs = [objs[i] for i in range(len(objs)) if mask[i]]
    for i in range(len(objs)):
        if mask[i]:
            objs[i].select()
            objs[i].hide(False)
        else:
            objs[i].deselect()
            objs[i].hide(True)
    return selected_objs


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', nargs='?',
                        default="assets/chairs",
                        help="Path to the object folder.")
    parser.add_argument('--texture', nargs='?',
                        default="assets/texture",
                        help="Path to the texture folder.")
    parser.add_argument('--output_dir', nargs='?', default="random_backgrounds/output_test",
                        help="Path to where the final files, will be saved")
    parser.add_argument('--average_object_per_image', default=3, type=int,
                        help="average number of objects per image")
    parser.add_argument('--config', type=str, help="path to config file")
    return parser


def preprocess_and_scale_objs(objs, category_id=1, name="chair", bbox_size=1):
    for obj in objs:
        set_label(category_id, name, obj)
        scale_obj(obj, bbox_size)


def scale_obj(obj, bbox_size):
    bbox_vol = obj.get_bound_box_volume()
    factor = bbox_size*(bbox_vol ** (-1. / 3.))
    obj_scale = obj.get_scale()
    obj.set_scale(obj_scale * factor)


def set_label(category_id, name, obj):
    obj.set_cp("category_id", category_id)
    obj.set_name(name)
    obj.enable_rigidbody(active=True, collision_shape="COMPOUND")
