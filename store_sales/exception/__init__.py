import os
import sys


class ApplicationException(Exception):
    def __init__(self, error_message: Exception, error_details:sys):
        self.error_message = ApplicationException.get_detailed_error_message(error_message=error_message, error_details=error_details)
        #self.error_details = error_details


    def get_detailed_error_message(error_message: Exception, error_details:sys)-> str:
        _,_, exec_tb = error_details.exc_info()
        Exception_block_line_number = exec_tb.tb_frame.f_lineno
        try_block_line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename
        error_message = f"""
                        error occured in script: 
                        [{file_name}] at 
                        try block line number: [{try_block_line_number}] execption block line number: [{Exception_block_line_number}]
                        error message: [{error_message}]"""
        

        return error_message
    
    def __str__(self):
        return self.error_message

    def __repr__(self) -> str:
        return ApplicationException.__name__.str()