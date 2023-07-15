import traceback
import sys

class ApplicationException(Exception):
    
    def __init__(self, error_message: Exception, error_details: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_details)

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_details: sys) -> str:
        """
        error_message: Exception object
        error_details: object of sys module
        """

        error_traceback = traceback.format_exc()
        error_type = type(error_message).__name__
        error_message = str(error_message)

        detailed_error_message = f"""
        Error type: {error_type}
        Error message: {error_message}
        Error traceback:
        {error_traceback}
        """

        return detailed_error_message.strip()

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return self.__class__.__name__