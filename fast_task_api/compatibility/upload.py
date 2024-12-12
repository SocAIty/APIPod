"""
UploadDataTypes are used to ensure compatibility between different providers. For example:
- fastapi has built in data types like "MediaFile", "File" which can be configured in detail and can be arbitrary.
- In runpod the job data always is json. Therefore, any upload data must be base64 encoded.
We will parse the data and always provide it as a binary object to your function.
"""
from inspect import Parameter
from media_toolkit import MediaFile, AudioFile, ImageFile, VideoFile
from starlette.datastructures import UploadFile as StarletteUploadFile


def _print_import_warning(class_name: str, lib_names: list):
    print(f"Necessary libraries: {', '.join(lib_names)} are not installed. "
          f"Please install them before using the {class_name} class.")

def is_param_media_toolkit_file(param: Parameter):
    """
    Check if a parameter is a file upload.
    """
    if param is None:
        return False

    from fastapi import UploadFile as fastapiUploadFile
    type_check_list = [
        MediaFile, ImageFile, AudioFile, VideoFile,
        StarletteUploadFile, fastapiUploadFile
    ]
    if not hasattr(param, 'annotation'):
        if not any(isinstance(param, t) for t in type_check_list):
            return False
        else:
            return True

    # check for class names also. This is the case if coming from a partial function instead of a real one
    if hasattr(param.annotation, '__name__'):
        return param.annotation.__name__ in [t.__name__ for t in type_check_list]

    return type(param.annotation) in type_check_list or param.annotation in type_check_list


def convert_param_type_to_fast_api_upload_file(param: Parameter):
    """
    Convert a UploadDataType to a FastAPI MediaFile type.
    """
    from fastapi import UploadFile as fastapiUploadFile
    return param.replace(annotation=fastapiUploadFile)
