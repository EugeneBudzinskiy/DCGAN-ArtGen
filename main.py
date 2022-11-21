import os
import sys
import time
import threading
import subprocess
import numpy as np
import tensorflow as tf

from PIL import Image
from typing import Union
from utils.validators import Validators

# noinspection PyUnresolvedReferences
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QTextCursor

MODELS_PATH = "models"

IMAGE_EXTENSION = "jpg"
OUTPUT_FOLDER_PATH = "output"

REAL_ESRGAN_PATH = "real-esrgan"
REAL_ESRGAN_INPUT_FOLDER = "input"
REAL_ESRGAN_OUTPUT_FOLDER = "output"
REAL_ESRGAN_MODEL = "realesrgan-x4plus"
REAL_ESRGAN_EXE = "realesrgan-ncnn-vulkan.exe"


class Window(QMainWindow):
    MIN_QUANTITY = 1
    MAX_QUANTITY = 100

    update_console = pyqtSignal(str)
    fully_finished = pyqtSignal()
    partially_finished = pyqtSignal()

    def __init__(self):
        super(Window, self).__init__()
        uic.loadUi('utils/design.ui', self)  # Load the .ui file

        self.available_models = self.list_all_folders(path=MODELS_PATH)
        # noinspection PyUnresolvedReferences
        self.generator_type.addItems(self.available_models)
        self.current_generator_type = self.available_models[0]
        # noinspection PyUnresolvedReferences
        self.generator_type.setCurrentText(self.current_generator_type)

        # noinspection PyUnresolvedReferences
        self.quantity.setText(f"{self.MIN_QUANTITY}")

        self.scale_option_name = ["64 x 64", "256 x 256", "1024 x 1024"]
        # noinspection PyUnresolvedReferences
        self.image_size.addItems(self.scale_option_name)
        self.current_scale_option = 0
        # noinspection PyUnresolvedReferences
        self.image_size.setCurrentText(
            self.scale_option_name[self.current_scale_option])

        self.current_quantity = None
        self.current_start_time = None
        self.current_seed = None
        self.current_output_path = None

        self.signal_logic()

    @staticmethod
    def list_all_folders(path: str) -> list:
        return [x for x in os.listdir(path) if os.path.isdir(f"{path}/{x}")]

    @staticmethod
    def remove_all_files_from_folder(path: str):
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))

    @staticmethod
    def show_validation_error(error_header: str = "Header", error_text: str = "Error text"):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Error")
        msg.setText(error_header)
        msg.setInformativeText(error_text)
        msg.exec_()

    # noinspection PyUnresolvedReferences
    def toggle_random_seed_field(self, state):
        if state:
            self.random_seed.setReadOnly(True)
            self.random_seed.setCursor(Qt.ForbiddenCursor)
            self.random_seed.setText("")
        else:
            self.random_seed.setReadOnly(False)
            self.random_seed.setCursor(Qt.IBeamCursor)

    # noinspection PyUnresolvedReferences
    def change_generator_type(self, index):
        self.current_generator_type = self.generator_type.itemText(index)

    # noinspection PyUnresolvedReferences
    def change_image_size(self, index):
        self.current_scale_option = index

    # noinspection PyUnresolvedReferences
    def get_and_validate_seed(self) -> Union[int, None]:
        if self.use_random_seed.isChecked():
            return np.random.randint(low=0, high=Validators.max_value)

        else:
            header = "Random seed validation error!"

            random_seed_raw = self.random_seed.text()
            res, msg = Validators.validate_input_len(random_seed_raw)
            if not res:
                self.show_validation_error(error_header=header, error_text=msg)
                return None

            res, msg = Validators.validate_int(random_seed_raw)
            if not res:
                self.show_validation_error(error_header=header, error_text=msg)
                return None

            random_seed_raw = int(random_seed_raw)
            res, msg = Validators.validate_value_overflow(random_seed_raw)
            if not res:
                self.show_validation_error(error_header=header, error_text=msg)
                return None

            return random_seed_raw

    # noinspection PyUnresolvedReferences
    def get_and_validate_quantity(self) -> Union[int, None]:
        header = "Quantity validation error!"

        quantity_raw = self.quantity.text()
        res, msg = Validators.validate_input_len(quantity_raw)
        if not res:
            self.show_validation_error(error_header=header, error_text=msg)
            return None

        res, msg = Validators.validate_int(quantity_raw)
        if not res:
            self.show_validation_error(error_header=header, error_text=msg)
            return None

        quantity_raw = int(quantity_raw)
        res, msg = Validators.validate_value_in_range(
            data=quantity_raw, low=self.MIN_QUANTITY, high=self.MAX_QUANTITY)
        if not res:
            self.show_validation_error(error_header=header, error_text=msg)
            return None

        return quantity_raw

    def add_log_file(self, path: str, seed: int, num: int, total_time: float):
        with open(f"{path}/log.txt", mode='w') as f:
            f.writelines([
                f"generator: {self.current_generator_type}\n",
                f"seed: {seed}\n",
                f"image number: {num}\n",
                f"scale option: {self.current_scale_option}\n",
                f"scale model: {REAL_ESRGAN_MODEL}\n",
                f"extension: {IMAGE_EXTENSION}\n",
                f"time per image (s): {total_time / num}\n",
                f"total run time (s): {total_time}\n"
            ])

    @staticmethod
    def generate_raw_images(generator: tf.keras.Sequential, num: int = 1) -> np.ndarray:
        input_shape = list(generator.input_shape)
        input_shape[0] = num
        gen_images = generator(tf.random.normal(input_shape)).numpy()
        return (255 * (gen_images + 1) / 2).astype("uint8")

    @staticmethod
    def load_generator(name: str) -> tf.keras.Sequential:
        return tf.keras.models.load_model(f"{MODELS_PATH}/{name}", compile=False)

    # noinspection PyUnresolvedReferences
    def add_line_to_console(self, line: str):
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(f"{line}\n")
        self.console.moveCursor(QTextCursor.End)

    # noinspection PyUnresolvedReferences
    def add_line_to_console_from_stdout(self, process: subprocess.Popen):
        for line in process.stdout:
            self.update_console.emit(f"    {str(line).strip()}")
        self.fully_finished.emit()

    # noinspection PyUnresolvedReferences
    def add_line_to_console_from_stdout_partial(self, process: subprocess.Popen):
        for line in process.stdout:
            self.update_console.emit(f"    {str(line).strip()}")
        self.partially_finished.emit()

    @staticmethod
    def get_command_in_out_folders() -> (str, str, str):
        abs_path = os.path.join(os.getcwd(), REAL_ESRGAN_PATH)
        e_gan_exe = os.path.join(abs_path, REAL_ESRGAN_EXE)
        input_folder = os.path.join(abs_path, REAL_ESRGAN_INPUT_FOLDER)
        output_folder = os.path.join(abs_path, REAL_ESRGAN_OUTPUT_FOLDER)
        command = f'"{e_gan_exe}" -i "{{i}}" -o "{{o}}" -n {REAL_ESRGAN_MODEL}'
        return command, input_folder, output_folder

    def run_generation(self):
        self.current_seed = self.get_and_validate_seed()
        self.current_quantity = self.get_and_validate_quantity()

        if self.current_seed is None or self.current_quantity is None:
            return

        self.add_line_to_console("\n[Info]: Starting..")
        tf.random.set_seed(self.current_seed)
        generator = self.load_generator(self.current_generator_type)

        self.current_start_time = time.time()
        self.add_line_to_console("[Info]: Generation 64x64 images..")
        gen_images = self.generate_raw_images(generator=generator, num=self.current_quantity)
        self.add_line_to_console("[Info]: Generation finished!")

        t = time.localtime()
        curr_time = f"{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}_" \
                    f"{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}"
        output_path = f"{OUTPUT_FOLDER_PATH}/{curr_time}"
        os.mkdir(output_path)
        self.current_output_path = output_path

        if self.current_scale_option == 1 or self.current_scale_option == 2:
            command, input_folder, output_folder = self.get_command_in_out_folders()

            for i in range(self.current_quantity):
                img = Image.fromarray(gen_images[i])
                img.save(f"{input_folder}/{i + 1:03d}.{IMAGE_EXTENSION}")

            if self.current_scale_option == 1:
                self.add_line_to_console("[Info]: Upscaling..")
                p = subprocess.Popen(args=command.format(i=input_folder, o=output_path),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                th = threading.Thread(target=self.add_line_to_console_from_stdout, args=(p,))
                th.start()

            elif self.current_scale_option == 2:
                self.add_line_to_console("[Info]: (1/2) Upscaling..")
                p = subprocess.Popen(args=command.format(i=input_folder, o=output_folder),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                th = threading.Thread(target=self.add_line_to_console_from_stdout_partial, args=(p,))
                th.start()

        else:
            for i in range(self.current_quantity):
                img = Image.fromarray(gen_images[i])
                img.save(f"{output_path}/{i + 1:03d}.{IMAGE_EXTENSION}")
            self.add_program_tail_to_console()

    def second_upscaling(self):
        command, _, output_folder = self.get_command_in_out_folders()
        self.add_line_to_console("[Info]: (2/2) Upscaling..")
        p = subprocess.Popen(args=command.format(i=output_folder, o=self.current_output_path),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        th = threading.Thread(target=self.add_line_to_console_from_stdout, args=(p,))
        th.start()

    def add_program_tail_to_console(self):
        _, input_folder, output_folder = self.get_command_in_out_folders()
        self.add_line_to_console("[Info]: Images saved!")

        elapsed_time = time.time() - self.current_start_time
        t_per_image = elapsed_time / self.current_quantity

        # Remove all temp images
        self.remove_all_files_from_folder(path=input_folder)
        self.remove_all_files_from_folder(path=output_folder)

        self.add_line_to_console("[Info]: Saving log file..")
        self.add_log_file(path=self.current_output_path, seed=self.current_seed,
                          num=self.current_quantity, total_time=elapsed_time)
        self.add_line_to_console("[Info]: Finishing..")
        self.add_line_to_console(f"[Info]: Time per image: {round(t_per_image, 5)} s")
        self.add_line_to_console(f"[Info]: Total run time: {round(elapsed_time, 5)} s")

    # noinspection PyUnresolvedReferences
    def signal_logic(self):
        self.update_console.connect(self.add_line_to_console)
        self.fully_finished.connect(self.add_program_tail_to_console)
        self.partially_finished.connect(self.second_upscaling)

        self.random_seed.setReadOnly(self.use_random_seed.checkState())
        self.use_random_seed.stateChanged.connect(self.toggle_random_seed_field)
        self.toggle_random_seed_field(True)

        self.generator_type.activated.connect(self.change_generator_type)
        self.image_size.activated.connect(self.change_image_size)

        self.run_button.released.connect(self.run_generation)


class Interface:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = Window()

    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())


def main():
    ui = Interface()
    ui.run()


if __name__ == '__main__':
    main()
