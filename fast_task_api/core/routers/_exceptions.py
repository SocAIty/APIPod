
class JobException(Exception):
    """
    If you raise an JobException the global exception handler will catch it and return a a failed job result.
    Only use this exception for errors that are not related to the job function.
    For example a file upload failed or other problem.
    This prevents causing an internal server error.
    """
    pass


class FileUploadException(JobException):
    def __init__(self, file_name: str = "", message: str = "File upload failed"):
        self.file_name = file_name

        if file_name:
            message = f"FileUploadException: {file_name} - {message}"
        else:
            message = f"FileUploadException: {message}"

        super().__init__(message)


class InsufficientBalanceException(JobException):
    """ Gets raised when a job is rejected because the user has insufficient balance """
    def __init__(self, message: str = "Insufficient balance"):
        super().__init__(message)
