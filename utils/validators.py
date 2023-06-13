import re

from typing import Union


class Validators:
    """
        A collection of static methods for data validation.

        This class provides various static methods for validating different types of data, such as integers
        and input lengths.

        Attributes:
            max_value (int): The maximum allowed value for validation.
            min_value (int): The minimum allowed value for validation.
            max_input_len (int): The maximum allowed length for input validation.
            re_int (str): The regular expression pattern for validating integers.
            err_suffix (str): The error suffix message used in error messages.
            err_wrong_type (str): The error message for incorrect data type.
            err_range (str): The error message for values outside the specified range.
            err_long_input (str): The error message for input exceeding the maximum length.
            err_out_range (str): The error message for values outside the variable's range.
    """

    # Variable allowed sizes
    max_value = 2 ** 31 - 1
    min_value = 0
    max_input_len = 20

    # Some Regular Expressions
    re_int = '^[0-9]+$'

    # Error massages
    err_suffix = 'Try again!'
    err_wrong_type = 'Wrong data type.'
    err_range = 'Value should be in range: '
    err_long_input = f'Hit the character limit. Max `{max_input_len}` char.'
    err_out_range = 'Value doesnt fit in variable.'

    @staticmethod
    def validate_value_overflow(data: Union[int, float]) -> tuple[bool, str]:
        """
            Validate if the value fits within the variable's range.

            Args:
                data (Union[int, float]): The value to validate.

            Returns:
                tuple[bool, str]: A tuple containing the validation result (True if within range, False otherwise)
                and an error message.
        """
        msg = ""
        result = Validators.min_value <= data <= Validators.max_value
        if not result:
            msg = f"{Validators.err_out_range} Should be in range " \
                  f"`{Validators.min_value}` to `{Validators.max_value}`"
        return result, msg

    @staticmethod
    def validate_value_in_range(data: Union[int, float], low: Union[int, float],
                                high: Union[int, float]) -> tuple[bool, str]:
        """
            Validate if the value is within the specified range.

            Args:
                data (Union[int, float]): The value to validate.
                low (Union[int, float]): The lower bound of the range.
                high (Union[int, float]): The upper bound of the range.

            Returns:
                tuple[bool, str]: A tuple containing the validation result (True if within range, False otherwise)
                and an error message.
        """
        msg = ""
        if not (low or high) and low != 0 and high != 0:
            return True, msg  # Return always True if borders not specified

        low, high = (low, high) if low and high else (0, low if low else high)
        if low > high:
            msg = f"Lower bound grated than higher bound (LOW={low} > HIGH={high})"
            return False, msg

        result = low <= data <= high
        if not result:
            msg = f"{Validators.err_range} {low} - {high}"
        return result, msg

    @staticmethod
    def validate_input_len(data: str) -> tuple[bool, str]:
        """
            Validate if the input length is within the specified limit.

            Args:
                data (str): The input data to validate.

            Returns:
                tuple[bool, str]: A tuple containing the validation result (True if within limit, False otherwise)
                and an error message.
        """
        msg = ""
        result = len(data) <= Validators.max_input_len
        if not result:
            msg = f"{Validators.err_long_input} {Validators.err_suffix}"
        return result, msg

    @staticmethod
    def validate_int(data: str) -> tuple[bool, str]:
        """
            Validate if the data is a positive integer.

            Args:
                data (str): The data to validate.

            Returns:
                tuple[bool, str]: A tuple containing the validation result (True if a positive int, False otherwise)
                and an error message.
        """
        msg = ""
        result = bool(re.match(Validators.re_int, data))
        if not result:
            msg = f"{Validators.err_wrong_type} Should be positive " \
                  f"integer. {Validators.err_suffix}"
        return result, msg
