#!/bin/bash
# bash random_backgrounds/generation_pipeline.sh  object_foler_path  output_dir_for_synthetic_images num num_camera background_folder
#      output_background project create output split

bash random_backgrounds/generation_pipeline.sh assets/chairs random_backgrounds/output_test 1 1 random_backgrounds/backgrounds \
     random_backgrounds/images_with_backgrounds_test/coco_data test true ./test 0.7 0.15 0.15

# bash random_backgrounds/generation_pipeline.sh assets/chairs random_backgrounds/output_test 1 1 random_backgrounds/backgrounds \
#      random_backgrounds/images_with_backgrounds_test/coco_data no
