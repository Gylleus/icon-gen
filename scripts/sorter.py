from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import click
import os
import sys


class LabelWindow(QWidget):
    def __init__(self, base_dir: str, classes):
        super().__init__()
        self.image_size = 192
        self.row = 0
        self.col = 0
        self.base_dir = base_dir
        self.images = self.image_files()
        self.initUI(classes)

    def next_image(self):
        self.current_image = next(self.images)
        pixmap = QPixmap(self.current_image)
        pixmap = pixmap.scaledToWidth(self.image_size).scaledToHeight(self.image_size)
        self.image_label.setPixmap(pixmap)
        self.image_name_label.setText(self.current_image)

    def image_files(self):
        for file in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, file)
            if os.path.isfile(path):
                yield path

    def add_button(self, c: str, hbox):
        btn = QPushButton(c, self)
        btn.setMaximumWidth(128)
        btn.clicked.connect(lambda: self.label_image(c))
        hbox.addWidget(btn, self.row + 1, self.col)
        self.col += 1
        if self.col % 3 == 0:
            self.row += 1
            self.col = 0

    def initUI(self, classes):
        imageNameHBox = QHBoxLayout()
        imageHBox = QHBoxLayout()
        buttonHBox = QGridLayout()
        # buttonHBox.columnMinimumWidth(100)0
        imageNameHBox.setAlignment(Qt.AlignCenter)
        imageHBox.setAlignment(Qt.AlignCenter)
        # buttonHBox.setAlignment(Qt.AlignCenter)

        self.image_name_label = QLabel(self)
        self.image_label = QLabel(self)
        self.image_name_label.setAlignment(Qt.AlignCenter)
        self.image_label.setAlignment(Qt.AlignCenter)
        imageNameHBox.addWidget(self.image_name_label)
        imageHBox.addWidget(self.image_label)
        for c in classes:
            self.add_button(c, buttonHBox)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(imageNameHBox)
        vbox.addLayout(imageHBox)
        vbox.addLayout(buttonHBox)

        self.setLayout(vbox)
        self.move(300, 300)
        self.setWindowTitle("Icons")
        self.next_image()

        self.show()

    def label_image(self, cls: str):
        new_file = os.path.join(
            self.base_dir, cls, os.path.basename(self.current_image)
        )
        os.rename(self.current_image, new_file)
        print(f"Moved {self.current_image} to {new_file}")
        self.next_image()


@click.command()
@click.option("--dir", "-d", required=True)
def main(dir: str):
    classes = [x[0].split("\\")[-1] for x in os.walk(dir)]
    dir_name = dir.split("\\")[-1]
    classes.remove(dir_name)

    app = QApplication(sys.argv)
    ex = LabelWindow(dir, classes)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()