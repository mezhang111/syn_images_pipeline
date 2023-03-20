import blenderproc as bproc
import numpy as np
import argparse
import random
from pathlib import Path
from blenderproc.python.types.MeshObjectUtility import MeshObject
import os


# import pydevd_pycharm


# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


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
        mat = obj.get_materials()[0]
        mat.set_principled_shader_value("Specular", random.uniform(0, 1))
        mat.set_principled_shader_value("Roughness", random.uniform(0, 1))
        mat.set_principled_shader_value("Base Color", np.random.uniform([0, 0, 0, 1], [1, 1, 1, 1]))
        mat.set_principled_shader_value("Metallic", random.uniform(0, 1))


def sample_pose(_obj: bproc.types.MeshObject):
    _obj.set_location(np.random.uniform([-5, -5, 0], [5, 5, 12]))
    _obj.set_rotation_euler(bproc.sampler.uniformSO3(True, True, True))


def sample_light(_light):
    _light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    _light.set_location(np.random.uniform([-5, -5, 0], [5, 5, 12]))
    _light.set_energy(np.random.uniform(100, 1000))


def sample_camera(_poi):
    location = bproc.sampler.shell(center=[0, 0, 0],
                                   radius_min=3,
                                   radius_max=5,
                                   elevation_min=5,
                                   elevation_max=89)
    lookat_point = _poi + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(lookat_point - location,
                                                             inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    return cam2world_matrix


def get_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', nargs='?',
                        default="assets/chairs",
                        help="Path to the object folder.")
    parser.add_argument('--output_dir', nargs='?', default="random_backgrounds/output_test",
                        help="Path to where the final files, will be saved")
    parser.add_argument('--num', default=5, type=int,
                        help="The number of times the objects should be repositioned and rendered using multiple"
                             "camera poses.")
    parser.add_argument('--cameras', default=2, type=int, help="The number of camera poses per object pose")
    args = parser.parse_args()
    return args


def main():
    global args
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    bproc.init()
    bproc.camera.set_resolution(640, 640)
    # load the objects into the scene
    path = Path(args.scene)
    objs = []
    for f in path.iterdir():
        if f.is_file():
            if f.suffix == '.blend':
                mesh_objs = bproc.loader.load_blend(str(f))
                if len(mesh_objs) > 1:
                    mesh_objs[0].join_with_other_objects(mesh_objs[1:])
                    merged_obj = mesh_objs[0]
                    # for obj in mesh_objs[1:]:
                    #   obj.deselect()
                    #    obj.delete()
                    merged_obj.select()

                objs.append(mesh_objs[0])
            elif f.suffix == '.obj':
                objs.append(bproc.loader.load_obj(str(f)))

    # set class and name for coco annotations, Make the object actively participate in the physics simulation\
    for obj in objs:
        obj.set_cp("category_id", 1)
        obj.set_name("chair")
        obj.enable_rigidbody(active=True, collision_shape="COMPOUND")
    # Create a new light
    light = bproc.types.Light()
    light.set_type("POINT")
    # Enable transparency so the background becomes transparent
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

    for r in range(args.num):
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()
        # sample light
        sample_light(light)
        # randomly select a subset of objects
        mask = np.random.choice(a=[False, True], size=(len(objs),))
        selected_objs = [objs[i] for i in range(len(objs)) if mask[i]]
        for i in range(len(objs)):
            if mask[i]:
                objs[i].select()
            else:
                objs[i].deselect()

        if len(selected_objs) > 0:
            sample_material(selected_objs)
            # Sample the poses of all shapenet objects above the ground without any collisions in-between
            bproc.object.sample_poses(
                selected_objs,
                sample_pose_func=sample_pose
            )
            # bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20,
            #                                                  check_object_interval=1)

            # sample camerase for this pose
            poi = bproc.object.compute_poi(selected_objs)
            max_tries = 10000
            n_tries = 0
            max_cameras = args.cameras
            n_cameras = 0
            while n_tries < max_tries and n_cameras < max_cameras:
                cam2world_matrix = sample_camera(poi)
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
