"""
UploadDataTypes are used to ensure compatibility between different providers. For example:
- fastapi has built in data types like "MediaFile", "File" which can be configured in detail and can be arbitrary.
- In runpod the job data always is json. Therefore, any upload data must be base64 encoded.
We will parse the data and always provide it as a binary object to your function.
"""
from inspect import Parameter
from media_toolkit import MediaFile, AudioFile, ImageFile, VideoFile
try:
    from starlette.datastructures import UploadFile as StarletteUploadFile
except ImportError:
    StarletteUploadFile = None


def check_if_param_is_in_data_types(param: Parameter, type_check_list: list):
    if param is None or type_check_list is None:
        return False

    # check for annotations
    if not hasattr(param, 'annotation'):
        if not any(isinstance(param, t) for t in type_check_list) and not any(param == t for t in type_check_list):
            return False
        else:
            return True

    # check for class names also. This is the case if coming from a partial function instead of a real one
    if hasattr(param.annotation, '__name__'):
        return param.annotation.__name__ in [t.__name__ for t in type_check_list]

    return type(param.annotation) in type_check_list or param.annotation in type_check_list


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
    return check_if_param_is_in_data_types(param, type_check_list)
