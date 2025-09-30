from huggingface_hub import snapshot_download


def download_spider_thorax(root_folder: str):
    import subprocess
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="histai/SPIDER-thorax",
        repo_type="dataset",
        local_dir=root_folder,
        local_dir_use_symlinks=False,
    )

    # unpack the tar files
    # run cat spider-thorax.tar.* | tar -xvf -

    result = subprocess.run(
        f"cat {root_folder}/spider-thorax.tar.* | tar -xvf - -C {root_folder}",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract tar files \nError: {result.stderr}")

    # delete the tar files
    import glob
    import os

    tar_files = glob.glob(os.path.join(root_folder, "spider-thorax.tar.*"))
    for tar_file in tar_files:
        os.remove(tar_file)

    # Convert to imagenet format

    cmd_to_run = (
        f"python {root_folder}/scripts/convert_to_imagenet.py "
        f"--data_dir {root_folder}/SPIDER-thorax "
        f"--output_dir {root_folder}/center_crop "
        f"--context_size 1 --num_workers 16"
    )

    result = subprocess.run(cmd_to_run, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to run command: {cmd_to_run}\nError: {result.stderr}"
        )


def create_splits_spider_thorax(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the spider thorax dataset.

    :param base_folder: path to the main folder after converting to imagenet format.
    :param dataset_cfg: dataset-specific config.
    """

    import os
    import random

    import numpy as np

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import (check_dataset, create_few_shot_training_data,
                               get_data_from_folder_recursive, init_dict,
                               save_dict)

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict

    spider_thorax_data_splits = init_dict()

    # Retrieving all images and labels
    spider_thorax_folder = os.path.join(base_folder, "spider_thorax")
    (
        train_val_images,
        train_val_labels,
    ) = get_data_from_folder_recursive(
        spider_thorax_folder, "center_crop/train", dataset_cfg
    )
    # Sampling val samples
    train_val_images, train_val_labels = (
        np.array(train_val_images),
        np.array(train_val_labels),
    )

    # image names are folders/to/classname/<slide_id>/<image_id>.png
    # we want to assign images from a single slide to the same split

    slide_names = [item.split("/")[-2] for item in train_val_images]
    unique_slide_names = sorted(list(set(slide_names)))
    val_slide_names = random.sample(
        unique_slide_names, int(0.2 * len(unique_slide_names))
    )

    # Class 12 is only present on 4 slides in the spider_thorax dataset
    # (significantly less represented than other classes)
    # so we need to make sure to have one in val set and the 3 others in train set
    for slide_name in ["slide_0021", "slide_0048", "slide_0123", "slide_0178"]:
        # Looping over the only 4 slides containing class 12 samples
        if slide_name != "slide_0048" and slide_name in val_slide_names:
            val_slide_names.remove(slide_name)
        elif slide_name == "slide_0048" and slide_name not in val_slide_names:
            val_slide_names.append(slide_name)

    val_mask = np.array([slide_name in val_slide_names for slide_name in slide_names])

    (
        spider_thorax_data_splits["val"]["images"],
        spider_thorax_data_splits["val"]["labels"],
    ) = (
        train_val_images[val_mask].tolist(),
        train_val_labels[val_mask].tolist(),
    )
    (
        spider_thorax_data_splits["train"]["images"],
        spider_thorax_data_splits["train"]["labels"],
    ) = (
        train_val_images[~val_mask].tolist(),
        train_val_labels[~val_mask].tolist(),
    )
    # Test samples
    (
        spider_thorax_data_splits["test"]["images"],
        spider_thorax_data_splits["test"]["labels"],
    ) = get_data_from_folder_recursive(
        spider_thorax_folder, "center_crop/test", dataset_cfg
    )

    check_dataset(
        spider_thorax_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Few-shot training data
    spider_thorax_data_splits = create_few_shot_training_data(spider_thorax_data_splits)

    # Saving dict
    save_dict(
        spider_thorax_data_splits,
        os.path.join(base_folder, "data_splits", "spider_thorax.json"),
    )
