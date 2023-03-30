import blenderproc as bproc
import sys
sys.path.append('./')
from omegaconf import OmegaConf
import utils
from pathlib import Path
import os
import numpy as np

import pydevd_pycharm

pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


# material_sample, object random location, object random rotation,
# camera random location, camera random look at point, camera random rotation,

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





def main():
    args = utils.get_default_parser().parse_args()
    config = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    bproc.init()
    bproc.camera.set_resolution(640, 640)
    # load the objects into the scene
    objs = utils.load_objs(args.object)
    # set class and name for coco annotations, Make the object actively participate in the physics simulation\
    for obj in objs:
        obj.set_cp("category_id", 1)
        obj.set_name("chair")
        obj.enable_rigidbody(active=True, collision_shape="COMPOUND")
        bbox_vol = obj.get_bound_box_volume()
        factor = bbox_vol ** (-1. / 3.)
        obj_scale = obj.get_scale()
        obj.set_scale(obj_scale * factor)
    # Create a new light
    light = bproc.types.Light()
    light.set_type("POINT")
    # Enable transparency so the background becomes transparent
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
    materials = bproc.material.collect_all()
    paths = list(Path(args.texture).absolute().rglob("*.jpg"))
    paths += list(Path(args.texture).absolute().rglob("*.png"))
    texture = [bproc.loader.load_texture(path)[0] for path in paths]
    base_material = "Base_Material"
    bproc.material.create_material_from_texture("/home/mengtao/experiments/assets/texture/chair-cora-nat-n.jpg",
                                                base_material)
    utils.check_material(base_material, materials)
    for r in range(config.num_pose):
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()
        # randomly select a subset of objects
        p = [0.9 , 0.1]
        selected_objs = utils.random_select_obj(objs, p)
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
                sample_pose_func=generate_sample_pose_fn(config.object),
                objects_to_check_collisions=[],
                max_tries=5
            )
            if config.simulate_physics:
                bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20,
                                                                  check_object_interval=1)
            # sample camerase for this pose
            poi = bproc.object.compute_poi(selected_objs)
            max_tries = 10000
            n_tries = 0
            max_cameras = config.num_camera
            n_cameras = 0
            while n_tries < max_tries and n_cameras < max_cameras:
                cam2world_matrix = sample_camera(poi, config.camera)
                if not set(selected_objs).isdisjoint(bproc.camera.visible_objects(cam2world_matrix)):
                    bproc.camera.add_camera_pose(cam2world_matrix)
                    n_cameras += 1
                n_tries += 1
            '''
            # Only add camera pose if object is still visible
            for obj in objs:
                if obj in bproc.camera.visible_objects(cam2world_matrix):
                    bproc.camera.add_camera_pose(cam2world_matrix)
            '''

            # render and append to output

            # add segmentation masks (per class and per instance)
            data = bproc.renderer.render()
            bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                                instance_segmaps=data["instance_segmaps"],
                                                instance_attribute_maps=data["instance_attribute_maps"],
                                                colors=data["colors"],
                                                append_to_existing_output=True)



if __name__ == '__main__':
    main()
