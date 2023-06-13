import os


class Toolbox:
    @staticmethod
    def check_if_folder_exist_and_try_create(path: str) -> bool:
        """
            Checks if a folder exists at the given path and tries to create it if it doesn't exist.

            Args:
                path (str): The path to the folder.

            Returns:
                bool: True if the folder already exists, or if it was successfully created. False if an error
                      occurred while creating the folder.

            Example:
                >>> Toolbox.check_if_folder_exist_and_try_create('/path/to/folder')
                True
        """
        try:
            os.mkdir(path)
        except FileExistsError:
            return True
        return False

    @staticmethod
    def list_all_folders(path: str) -> list:
        """
            Returns a list of all folders in the specified path.

            Args:
                path (str): The path to search for folders.

            Returns:
                list: A list of folder names.

            Example:
                >>> Toolbox.list_all_folders('/path/to/directory')
                ['folder1', 'folder2', 'folder3']
        """
        return [x for x in os.listdir(path) if os.path.isdir(f"{path}/{x}")]

    @staticmethod
    def remove_all_files_from_folder(path: str):
        """
            Removes all files from a folder at the specified path.

            Args:
                path (str): The path to the folder.

            Example:
                >>> Toolbox.remove_all_files_from_folder('/path/to/folder')
        """
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))
