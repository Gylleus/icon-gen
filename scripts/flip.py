import os
import cv2
import click


@click.command()
@click.option("--dir", required=True)
@click.option("--vertical/--no-vertical", default=False)
@click.option("--horizontal/--no-horizontal", default=False)
def main(dir, vertical, horizontal):
    if not vertical and not horizontal:
        print("No changes to apply.")
        return
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        img = cv2.imread(file_path)
        print(file_path)
        if vertical:
            vertical_img = cv2.flip(img, 1)
            cv2.imwrite(f"{file_path}_flip_vertical.jpg", vertical_img)
        if horizontal:
            horizontal_img = cv2.flip(img, 0)
            cv2.imwrite(f"{file_path}_flip_horizontal.jpg", horizontal_img)
        if horizontal and vertical:
            vertical_horizontal_img = cv2.flip(cv2.flip(img, 1), 0)
            cv2.imwrite(
                f"{file_path}_flip_vertical_horizontal.jpg", vertical_horizontal_img
            )


if __name__ == "__main__":
    main()
