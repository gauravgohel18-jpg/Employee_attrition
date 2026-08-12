import sys

def error_message_detail(error, error_detail:sys):
    _,_, exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
    file_name,exc_tb.tb_lineno, str(error))

    return  (
            f"Error occurred in file: {file_name} "
            f"at line: {line_number} "
            f"Error message: {error_message}"
        )

class CustomException(Exception):

    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail (error_message, error_details)
        
    def _str_(self):
        return self.error_message
    