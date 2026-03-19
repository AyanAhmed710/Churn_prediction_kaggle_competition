import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message=error_message
        _ ,_ , exc_tb=error_detail.exc_info()

        self.file_name=exc_tb.tb_frame.f_code.co_filename
        self.line_number=exc_tb.tb_lineno

    def __str__(self):

        return f"Error occurred in file: {self.file_name} at line number: {self.line_number} with error message: {self.error_message}"  
    

    

    
        