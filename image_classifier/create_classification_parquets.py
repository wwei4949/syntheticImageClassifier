import pandas as pd
from pathlib import Path


def get_images_in_directory(directory, filetypes=(".jpg", ".png")):
    return [file.resolve().as_posix() for filetype in filetypes for file in Path(directory).rglob(f"*{filetype}")]


def train_val_split(df: pd.DataFrame, val_frac: float = 0.30, seed=None):
    if not (0 < val_frac < 1):
        raise Exception("val_frac is expected to be in the range (0,1)")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(val_frac * len(df))
    train = df[split_idx:]
    val = df[:split_idx]
    return train, val


if __name__ == "__main__":
    data_dir = "data"

    train_real_images = get_images_in_directory(f"{data_dir}/severe")
    train_fake_generated_images = get_images_in_directory(f"{data_dir}/train_val_fake_generated")
    train_fake_inpainted_images = get_images_in_directory(f"{data_dir}/train_val_fake_inpainting")
    train_fake_images = train_fake_generated_images + train_fake_inpainted_images

    test_real_images = get_images_in_directory(f"{data_dir}/curated")
    test_fake_generated_images = get_images_in_directory(f"{data_dir}/test_fake_generated")
    test_fake_inpainted_images = get_images_in_directory(f"{data_dir}/test_fake_inpainting")
    test_fake_images = test_fake_generated_images + test_fake_inpainted_images

    # 0 = real image
    # 1 = fake image

    train_val_df = pd.DataFrame({"img_path": train_real_images + train_fake_images,
                                 "gt_label": [0] * len(train_real_images) + [1] * len(train_fake_images)})
    train_df, val_df = train_val_split(train_val_df, val_frac=0.30, seed=0)
    train_df.to_parquet("train.parquet")
    val_df.to_parquet("val.parquet")
    print(f"train={len(train_df)}")
    print(f"val={len(val_df)}")

    test_df = pd.DataFrame({"img_path": test_real_images + test_fake_images,
                            "gt_label": [0] * len(test_real_images) + [1] * len(test_fake_images)})
    test_df.to_parquet("test.parquet")
    print(f"test={len(test_df)}")
