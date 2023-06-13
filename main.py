import os
import sys
import time
import threading
import subprocess
import numpy as np
import tensorflow as tf

from PIL import Image
from typing import Union
from utils.toolbox import Toolbox
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
    """
        Main window class for a PyQt application.
    """
    MIN_QUANTITY = 1
    MAX_QUANTITY = 100

    update_console = pyqtSignal(str)
    fully_finished = pyqtSignal()
    partially_finished = pyqtSignal()

    def __init__(self):
        """
            Initialize the Window class.
        """
        super(Window, self).__init__()
        uic.loadUi('utils/design.ui', self)  # Load the .ui file

        # create necessary folder for upscaling by REAL-ESRGAN
        Toolbox.check_if_folder_exist_and_try_create(path=os.path.join(REAL_ESRGAN_PATH, REAL_ESRGAN_INPUT_FOLDER))
        Toolbox.check_if_folder_exist_and_try_create(path=os.path.join(REAL_ESRGAN_PATH, REAL_ESRGAN_OUTPUT_FOLDER))

        self.available_models = Toolbox.list_all_folders(path=MODELS_PATH)
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
    def show_validation_error(error_header: str = "Header", error_text: str = "Error text"):
        """
            Show a validation error message box.

            Args:
                error_header (str, optional): The header text of the error message box. Defaults to "Header".
                error_text (str, optional): The body text of the error message box. Defaults to "Error text".
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Error")
        msg.setText(error_header)
        msg.setInformativeText(error_text)
        msg.exec_()

    # noinspection PyUnresolvedReferences
    def toggle_random_seed_field(self, state):
        """
            Toggle the random seed field based on the checkbox state.

            Args:
                state: The state of the "use_random_seed" checkbox.
        """
        if state:
            self.random_seed.setReadOnly(True)
            self.random_seed.setCursor(Qt.ForbiddenCursor)
            self.random_seed.setText("")
        else:
            self.random_seed.setReadOnly(False)
            self.random_seed.setCursor(Qt.IBeamCursor)

    # noinspection PyUnresolvedReferences
    def change_generator_type(self, index):
        """
            Change the current generator type based on the selected index.

            Args:
                index: The index of the selected generator type.
        """
        self.current_generator_type = self.generator_type.itemText(index)

    # noinspection PyUnresolvedReferences
    def change_image_size(self, index):
        """
            Change the current image size based on the selected index.

            Args:
                index: The index of the selected image size.
        """
        self.current_scale_option = index

    # noinspection PyUnresolvedReferences
    def get_and_validate_seed(self) -> Union[int, None]:
        """
            Get and validate the random seed input.

            Returns:
                Union[int, None]: The validated random seed value, or None if validation fails.
        """
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
        """
            Get and validate the quantity input.

            Returns:
                Union[int, None]: The validated quantity value, or None if validation fails.
        """
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
        """
            Add a log file with generation information.

            Args:
                path (str): The path to save the log file.
                seed (int): The random seed used for generation.
                num (int): The number of images generated.
                total_time (float): The total generation time.
        """
        with open(f"{path}/log.txt", mode='w') as f:
            f.writelines([
                f"generator: {self.current_generator_type}\n",
                f"seed: {seed}\n",
                f"image number: {num}\n",
                f"scale option: {self.current_scale_option} ({self.scale_option_name[self.current_scale_option]})\n",
                f"scale model: {REAL_ESRGAN_MODEL}\n",
                f"extension: {IMAGE_EXTENSION}\n",
                f"time per image (s): {total_time / num}\n",
                f"total run time (s): {total_time}\n"
            ])

    @staticmethod
    def generate_raw_images(generator: tf.keras.Sequential, num: int = 1) -> np.ndarray:
        """
            Generate raw images using the given generator.

            Args:
                generator (tf.keras.Sequential): The generator model.
                num (int, optional): The number of images to generate. Defaults to 1.

            Returns:
                np.ndarray: The generated raw images.
        """
        input_shape = list(generator.input_shape)
        input_shape[0] = num
        gen_images = generator(tf.random.normal(input_shape)).numpy()
        return (255 * (gen_images + 1) / 2).astype("uint8")

    @staticmethod
    def load_generator(name: str) -> tf.keras.Sequential:
        """
            Load a generator model with the given name.

            Args:
                name (str): The name of the generator model.

            Returns:
                tf.keras.Sequential: The loaded generator model.
        """
        return tf.keras.models.load_model(f"{MODELS_PATH}/{name}", compile=False)

    # noinspection PyUnresolvedReferences
    def add_line_to_console(self, line: str):
        """
            Add a line to the console.

            Args:
                line (str): The line to be added.
        """
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(f"{line}\n")
        self.console.moveCursor(QTextCursor.End)

    # noinspection PyUnresolvedReferences
    def add_line_to_console_from_stdout(self, process: subprocess.Popen):
        """
            Add lines to the console from the stdout of a subprocess.

            Args:
                process (subprocess.Popen): The subprocess.
        """
        for line in process.stdout:
            self.update_console.emit(f"    {str(line).strip()}")
        self.fully_finished.emit()

    # noinspection PyUnresolvedReferences
    def add_line_to_console_from_stdout_partial(self, process: subprocess.Popen):
        """
            Add lines to the console from the stdout of a sub-subprocess.

            Args:
                process (subprocess.Popen): The subprocess.
        """
        for line in process.stdout:
            self.update_console.emit(f"    {str(line).strip()}")
        self.partially_finished.emit()

    @staticmethod
    def get_command_in_out_folders() -> (str, str, str):
        """
            Get the command and input/output folders for the Real-ESRGAN subprocess.

            Returns:
                Tuple[str, str, str]: The command, input folder, and output folder.
        """
        abs_path = os.path.join(os.getcwd(), REAL_ESRGAN_PATH)
        e_gan_exe = os.path.join(abs_path, REAL_ESRGAN_EXE)
        input_folder = os.path.join(abs_path, REAL_ESRGAN_INPUT_FOLDER)
        output_folder = os.path.join(abs_path, REAL_ESRGAN_OUTPUT_FOLDER)
        command = f'"{e_gan_exe}" -i "{{i}}" -o "{{o}}" -n {REAL_ESRGAN_MODEL}'
        return command, input_folder, output_folder

    def run_generation(self):
        """
            Run the image generation process.
        """
        self.current_seed = self.get_and_validate_seed()
        self.current_quantity = self.get_and_validate_quantity()

        if self.current_seed is None or self.current_quantity is None:
            return

        self.add_line_to_console("\n[INFO]: Starting..")
        tf.random.set_seed(self.current_seed)
        generator = self.load_generator(self.current_generator_type)

        self.current_start_time = time.time()
        self.add_line_to_console("[INFO]: Generation 64x64 images..")
        gen_images = self.generate_raw_images(generator=generator, num=self.current_quantity)
        self.add_line_to_console("[INFO]: Generation finished!")

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
                self.add_line_to_console("[INFO]: Upscaling..")
                p = subprocess.Popen(args=command.format(i=input_folder, o=output_path),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                th = threading.Thread(target=self.add_line_to_console_from_stdout, args=(p,))
                th.start()

            elif self.current_scale_option == 2:
                self.add_line_to_console("[INFO]: (1/2) Upscaling..")
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
        self.add_line_to_console("[INFO]: (2/2) Upscaling..")
        p = subprocess.Popen(args=command.format(i=output_folder, o=self.current_output_path),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        th = threading.Thread(target=self.add_line_to_console_from_stdout, args=(p,))
        th.start()

    def add_program_tail_to_console(self):
        _, input_folder, output_folder = self.get_command_in_out_folders()
        self.add_line_to_console("[INFO]: Images saved!")

        elapsed_time = time.time() - self.current_start_time
        t_per_image = elapsed_time / self.current_quantity

        # Remove all temp images
        Toolbox.remove_all_files_from_folder(path=input_folder)
        Toolbox.remove_all_files_from_folder(path=output_folder)

        self.add_line_to_console("[INFO]: Saving log file..")
        self.add_log_file(path=self.current_output_path, seed=self.current_seed,
                          num=self.current_quantity, total_time=elapsed_time)
        self.add_line_to_console("[INFO]: Finishing..")
        self.add_line_to_console(f"[INFO]: Time per image: {round(t_per_image, 5)} s")
        self.add_line_to_console(f"[INFO]: Total run time: {round(elapsed_time, 5)} s")

    # noinspection PyUnresolvedReferences
    def signal_logic(self):
        """
            Connect signals and slots.
        """
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
    """
        The main interface for the application.

        This class initializes the application and creates an instance of the Window class.
        It provides a method to run the interface.

        Attributes:
            app (QApplication): The application object.
            window (Window): The main window of the application.
    """
    def __init__(self):
        """
            Initialize the Interface object.

            Creates an instance of QApplication and Window.
        """
        self.app = QApplication(sys.argv)
        self.window = Window()

    def run(self):
        """
            Run the application.

            Shows the main window and starts the application event loop.
        """
        self.window.show()
        sys.exit(self.app.exec_())


def main():
    # Main process
    ui = Interface()
    ui.run()


if __name__ == '__main__':
    main()
