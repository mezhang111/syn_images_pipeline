import blenderproc as bproc
import sys

sys.path.append('')
from omegaconf import OmegaConf
import utils
from pathlib import Path
import os
import numpy as np
import heapq
import json
import pandas as pd


# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


def _get_random_range(flags, random_ranges, fix_points):
    l = len(flags)
    res = [random_ranges[i] if flags[i] else fix_points[i] for i in range(l)]
    return res


def generate_sample_pose_fn(config, surface=None, face_sample_range=[0, 0.5], min_height=0, max_height=0.01):
    location_cfg = config.location.random
    [range_min, range_max] = utils.get_xyz_range(location_cfg, [[-3, 3]] * 3, [[-8, -8], [0, 0], [0, 0]])

    def sample_pose(_obj: bproc.types.MeshObject):
        if surface is None:
            _obj.set_location(np.random.uniform(range_min, range_max))
        else:
            _obj.set_location(bproc.sampler.upper_region(
                objects_to_sample_on=[surface],
                min_height=min_height,
                max_height=max_height,
                use_ray_trace_check=False,
                face_sample_range=face_sample_range
            ))
        rotation_delta = bproc.sampler.uniformSO3(config.rotation.random.x, config.rotation.random.y,
                                                  config.rotation.random.z)
        _obj.set_rotation_euler(_obj.get_rotation_euler() + rotation_delta)

    return sample_pose


def sample_camera(_poi, config, surface=None, face_sample_range=[0.55, 0.9], min_height=2.5, max_height=4):
    location_cfg = config.location.random
    [elevation, azimuth] = _get_random_range([location_cfg.elevation, location_cfg.azimuth], [[-45, 45], [-90, 90]],
                                             [[25, 25.1], [0, 0.01]])
    if surface is None:
        location = bproc.sampler.shell(center=[0, 0, 0],
                                       radius_min=3,
                                       radius_max=5,
                                       elevation_min=elevation[0],
                                       elevation_max=elevation[1],
                                       azimuth_min=azimuth[0],
                                       azimuth_max=azimuth[1])
    else:
        location = bproc.sampler.upper_region(
            objects_to_sample_on=[surface],
            min_height=min_height,
            max_height=max_height,
            use_ray_trace_check=False,
            face_sample_range=face_sample_range
        )
    look_at_cfg = config.look_at.random
    range_min, range_max = utils.get_xyz_range(look_at_cfg, [[-0.5, 0.5], [-0.5, 0.5], [0.3, 0.7]], [[0, 0]] * 3)
    lookat_point = _poi + np.random.uniform(range_min, range_max)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(lookat_point - location,
                                                             inplane_rot=np.random.uniform(-0.7854, 0.7854)
                                                             if config.in_plane_rotation.random else 0.0)
    # Add homog cam pose based on location a rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    return cam2world_matrix


def main():
    parser = utils.get_default_parser()
    parser.add_argument("--surface", type=str, default="../assets/surface/surface.blend", help="Path to surface blend")
    parser.add_argument("--store_metadata", action="store_true", help="store metadata along with you")
    parser.add_argument("--other_obj", type=str, required=False, help="Path to other objs")
    parser.add_argument("--light", type=str, help="Path to hanging lights")
    args = parser.parse_args()
    bproc.init()
    config = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    bproc.camera.set_resolution(640, 640)
    # load the objects into the scene
    objs, objs_names = utils.load_objs(args.object)
    # set class and name for coco annotations, Make the object actively participate in the physics simulation\
    utils.preprocess_and_scale_objs(objs, objs_names)
    if args.other_obj is not None:
        other_objs, other_objs_names = utils.load_objs(args.other_obj)
        utils.preprocess_and_scale_objs(other_objs, other_objs_names, category_id=0)
    else:
        other_objs = []
    if args.light is not None:
        lights, lights_names = utils.load_objs(args.light)
        utils.preprocess_and_scale_objs(lights, lights_names, category_id=0)
    else:
        lights = []
    # Create a new light
    light = bproc.types.Light()
    light.set_type("POINT")
    # Enable transparency so the background becomes transparent
    set_renderer()
    materials, texture = prepare_material_and_texture_from_folder(args.texture)
    p_select = min(args.average_object_per_image / len(objs), 1.)
    p_select_other = min(args.average_object_per_image / len(other_objs), 1.) if len(other_objs) > 0 else 0
    p_select_light = min(1 / len(lights), 1) if len(lights) > 0 else 0
    surface = prepare_surface(args.surface, args.average_object_per_image*3)
    output_dir = os.path.join(args.output_dir, 'coco_data')
    store_metadata = args.store_metadata
    if store_metadata:
        instance_path = os.path.join(output_dir, "instance.csv")
        depth_path = os.path.join(output_dir, "depth.npy")
        # if not os.path.isfile(metadata_path):
        #     os.makedirs(output_dir, exist_ok=True)
        #     df = pd.DataFrame(columns=col_names)
        #     df.to_csv(metadata_path, index=False)
    else:
        instance_path = depth_path = None
    for _ in range(config.num_pose):
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()
        # randomly select a subset of objects

        selected_objs = utils.random_select_obj(objs, p_select=p_select, p_not_select=max(0., 1. - p_select),
                                                max_objs=round(args.average_object_per_image * 1.3))
        selected_objs_other = utils.random_select_obj(other_objs, p_select=p_select_other,
                                                      p_not_select=max(0., 1. - p_select_other),
                                                      max_objs=round(args.average_object_per_image * 1.3))
        selected_lights = utils.random_select_obj(lights, p_select=p_select_light,
                                                    p_not_select=max(0., 1-p_select_light),
                                                    max_objs=3)
        n_cameras = 0
        selected_objs = selected_objs + selected_objs_other  # put those 2 together because they would be on the ground
        if len(selected_objs) > 0:
            utils.random_material_texture_infusion(materials, selected_objs, texture)
            # sample light
            if config.sample_light:
                utils.sample_light(light)
            if config.sample_material:
                utils.sample_material(selected_objs)
            # Sample the poses of all shapenet objects above the ground without any collisions in-between
            bproc.object.sample_poses(
                selected_objs,
                sample_pose_func=generate_sample_pose_fn(config.object, surface=surface),
                objects_to_check_collisions=[],
                max_tries=10
            )
            if len(selected_lights) > 0:
                utils.random_material_texture_infusion(materials, selected_lights, texture)
                if config.sample_material:
                    utils.sample_material(selected_lights)
                bproc.object.sample_poses(
                    selected_lights,
                    sample_pose_func=generate_sample_pose_fn(config.object, surface=surface,
                                                             min_height=3, max_height=5, face_sample_range=[0, 0.5]),
                    objects_to_check_collisions=[],
                    max_tries=10
                )
            if config.simulate_physics:
                bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20,
                                                                  check_object_interval=1)
            # sample camerase for this pose
            poi = bproc.object.compute_poi(selected_objs)
            max_tries = 10000
            n_tries = 0
            max_cameras = config.num_camera
            while n_tries < max_tries and n_cameras < max_cameras:
                cam2world_matrix = sample_camera(poi, config.camera, surface=surface)
                if not set(selected_objs).isdisjoint(bproc.camera.visible_objects(cam2world_matrix)):
                    bproc.camera.add_camera_pose(cam2world_matrix)
                    n_cameras += 1
                n_tries += 1

        # render and append to output
        # add segmentation masks (per class and per instance)
        render_coco_and_write_metadata(instance_path, depth_path, n_cameras, output_dir, store_metadata)

    '''
    if store_metadata:
        df = pd.DataFrame(instance_info)
        np_depth = np.stack(depths, axis=0)
        if os.path.isfile(depth_path):
            old_depth = np.load(depth_path)
            np_depth = np.stack([old_depth, np_depth], axis=0)
        df.to_csv(instance_path, mode="a+")
        np.save(depth_path, np_depth)
    '''


def prepare_surface(surface_path, surface_size):
    if surface_path is not None:
        surface = bproc.loader.load_blend(str(surface_path))[0]
        # surface.set_rotation_euler(np.array([0, 0, 0]))
        surface.set_local2world_mat(np.identity(4))
        utils.scale_obj(surface, surface_size)
        surface.hide()
    else:
        surface = None
    return surface


def prepare_material_and_texture_from_folder(texture_path):
    materials = bproc.material.collect_all()
    paths = list(Path(texture_path).absolute().rglob("*.jpg"))
    paths += list(Path(texture_path).absolute().rglob("*.png"))
    paths += list(Path(texture_path).absolute().rglob("*.jpeg"))
    texture = [bproc.loader.load_texture(path)[0] for path in paths]
    base_material = "Base_Material"
    bproc.material.create_material_from_texture("../assets/texture/chair-cora-nat-n.jpg",
                                                base_material)
    utils.check_material(base_material, materials)
    return materials, texture


def set_renderer():
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"],
                                              default_values={"category_id": None, "name": None, "instance": None})
    bproc.renderer.enable_depth_output(activate_antialiasing=False)


def render_coco_and_write_metadata(instance_path, depth_path, n_cameras, output_dir, store_metadata):
    if n_cameras > 0:
        data = bproc.renderer.render()
        bproc.writer.write_coco_annotations(output_dir,
                                            instance_segmaps=data["instance_segmaps"],
                                            instance_attribute_maps=data["instance_attribute_maps"],
                                            colors=data["colors"],
                                            append_to_existing_output=True)
        if store_metadata:
            if not os.path.exists(instance_path) or not os.path.exists(depth_path):
                # If files don't exist, create new files and store the values
                instance_info = {'name': [], 'id': []}
                depths = []

                for i in range(n_cameras):
                    ids, names = get_names_and_ids(data, i)
                    depths.append(np.asarray(data['depth'][i], dtype=np.float16))
                    instance_info['name'].append(names)
                    instance_info['id'].append(ids)

                # Save instance_info to CSV
                df = pd.DataFrame(instance_info)
                df.to_csv(instance_path, index=False)
                # Save depths to NPY
                np.save(depth_path, depths)
            else:
                # If files exist, append the values
                instance_info = {'name': [], 'id': []}
                depths = np.load(depth_path)

                for i in range(n_cameras):
                    ids, names = get_names_and_ids(data, i)
                    curr_depth_map = np.expand_dims(np.asarray(data['depth'][i], dtype=np.float16), axis=0)
                    depths = np.concatenate([depths,  curr_depth_map], axis=0)
                    instance_info['name'].append(names)
                    instance_info['id'].append(ids)

                # Save instance_info and depths
                df = pd.DataFrame(instance_info)
                df.to_csv(instance_path, mode='a', header=False, index=False)
                np.save(depth_path, depths)
            '''
            for i in range(n_cameras):
                names = [instance['name'] for instance in data["instance_attribute_maps"][i][1:]]
                ids = [instance['idx'] for instance in data["instance_attribute_maps"][i][1:]]
                depths.append(np.asarray(data['depth'][i], dtype=np.float16))
                instance_info['name'] += names
                instance_info['id'] += ids
            '''


def get_names_and_ids(data, i):
    names = [instance['name'] for instance in data["instance_attribute_maps"][i][1:] if instance['category_id'] > 0]
    ids = [instance['idx'] for instance in data["instance_attribute_maps"][i][1:] if instance['category_id'] > 0]
    return ids, names


if __name__ == '__main__':
    main()
