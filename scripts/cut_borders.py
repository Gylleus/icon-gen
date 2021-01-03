import os
import cv2
import click
import shutil


@click.command()
@click.option("--dir", "-d", required=True)
@click.option("--out_dir", "-o", required=True)
@click.option("--border_width", "-w", required=True, type=int)
def main(dir: str, out_dir: str, border_width: int):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    for file in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, file))
        h, w, _ = img.shape
        cropped = img[border_width : h - border_width, border_width : w - border_width]

        # saving now
        file_path = os.path.join(out_dir, file)
        cv2.imwrite(f"{file_path}.jpg", cropped)


if __name__ == "__main__":
    main()
