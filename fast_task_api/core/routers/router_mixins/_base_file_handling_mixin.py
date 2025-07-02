import functools
import inspect
from types import UnionType
from typing import Any, Union, get_args, get_origin, Callable, List, Type

from media_toolkit import media_from_any, MediaFile, MediaList, MediaDict
from fast_task_api.compatibility.upload import is_param_media_toolkit_file
from fast_task_api.core.job.job_result import FileModel
from fast_task_api.core.routers._exceptions import FileUploadException


class _BaseFileHandlingMixin:
    """
    Base mixin for handling file uploads and parameter conversions across different deployment environments.

    Provides core functionality for:
    1. Identifying media-related parameters
    2. Supporting complex parameter types including optional and list parameters
    3. Flexible file conversion strategies
    """

    def __init__(self, max_upload_file_size_mb: float = None, *args, **kwargs):
        """
        Initialize the FileHandlingMixin.

        Args:
            max_upload_file_size_mb: Default maximum file size in MB for uploads
        """
        self.max_upload_file_size_mb = max_upload_file_size_mb

    def _is_media_param(self, annotation: Any) -> bool:
        """
        Determine if a parameter is a media-related type.
        This is different to _is_param_media_toolkit_file because it also checks for MediaList, Union, UnionType, List, list

        Args:
            annotation: Type annotation to check or inspect.Parameter object

        Returns:
            bool: True if the parameter is a media-related type
        """
        if annotation == MediaList:
            return True

        if inspect.isclass(annotation) and issubclass(annotation, FileModel):
            return True

        # Check for Union/UnionType with media file types included
        if get_origin(annotation) in [Union, UnionType, List, list]:
            return any(self._is_media_param(arg) for arg in get_args(annotation))

        # Direct media file type check
        return is_param_media_toolkit_file(annotation)

    def _get_media_target_type(self, annotation: Any) -> Type:
        """
        Determine the most appropriate MediaFile type for conversion.

        Args:
            annotation: Type annotation to extract media type from

        Returns:
            Type: Target MediaFile type

        Raises:
            ValueError: If invalid type combinations are detected
        """
        # Handle None/Any types
        if annotation is None or annotation == Any:
            return MediaFile

        # Handle Union/UnionType with multiple types
        org_annotation = get_origin(annotation)

        if org_annotation == MediaDict:
            raise ValueError("Use MediaList for declaring upload files instead of MediaDict")

        if org_annotation in [Union, UnionType]:
            args = get_args(annotation)

            # resolve recursively
            resolved_args = [self._get_media_target_type(arg) for arg in args]

            # if one of the args is a MediaList we need to treat it as a MediaList
            for arg in resolved_args:
                if arg == MediaList:
                    return arg

            # Handle Union with MediaFile types
            media_file_types = [t for t in args if is_param_media_toolkit_file(t)]
            if media_file_types and len(media_file_types) == 1:
                return media_file_types[0]

            # Return the first MediaFile type that's a specific FileType.
            media_file_types.sort(key=lambda x: x == MediaFile)
            return media_file_types[0]

        # Handle MediaList with generic type
        if org_annotation == MediaList:
            generic_type = get_args(annotation)
            if not generic_type:
                return MediaList

            # Check for nested MediaList
            if any(t == MediaList for t in generic_type):
                raise ValueError("Nesting of MediaList is not supported")

            # Extract specific MediaFile type if present
            media_list_types = [t for t in generic_type if is_param_media_toolkit_file(t)]
            if len(media_list_types) > 1:
                return MediaList
            elif len(media_list_types) == 1:
                return media_list_types[0]  # Deliver the first one with the specified generyc type

            return MediaList

        # Handle List types
        if org_annotation in [List, list]:
            args = get_args(annotation)

            if any(t == MediaList for t in args):
                raise ValueError("Nesting of MediaFiles List[MediaList] is not supported")

            if any(t == list and self._is_media_param(t) for t in args):
                raise ValueError("Nesting of MediaFiles List[List[MediaFile]] is not supported")

            # Check for MediaDict
            if any(t == MediaDict for t in args):
                raise ValueError("Use MediaList for declaring upload files instead of MediaDict")

            # Extract MediaFile types
            media_file_types = [t for t in args if is_param_media_toolkit_file(t)]
            if media_file_types and len(media_file_types) == 1:
                return MediaList[media_file_types[0]]
            return MediaList

        # Direct media file type
        if is_param_media_toolkit_file(annotation):
            return annotation

        return MediaFile

    def _convert_param_to_media_file(self, param_value: Any, annotation: Any) -> Any:
        """
        Convert a parameter to MediaFile, with fallback mechanisms.

        Args:
            param_value: Value to convert
            annotation: Type annotation guiding conversion

        Returns:
            Converted MediaFile or original value
        """
        # Skip conversion if not a media-related parameter
        if not self._is_media_param(annotation):
            return param_value

        try:
            # Determine target type for conversion
            target_type = self._get_media_target_type(annotation)

            # Attempt conversion
            m = media_from_any(
                data=param_value,
                type_hint=target_type,
                use_temp_file=True,
                temp_dir=None,
                allow_reads_from_disk=False
            )
            return m
        except Exception as e:
            # If strict conversion fails and it's a Union type, return original
            if get_origin(annotation) in [Union, UnionType]:
                return param_value

            # If conversion fails for a specific type, raise an error
            raise ValueError(f"Invalid upload file format: {str(e)}")

    def _sig_to_annotations(self, sig: Union[callable, inspect.Signature]) -> dict:
        """
        Convert a signature to a dictionary of parameter names and their annotations or get it from the function.
        """
        if callable(sig):
            sig = inspect.signature(sig)

        return {
            param.name: param.annotation if getattr(param, 'annotation', None) != inspect.Parameter.empty else Any
            for param in sig.parameters.values()
        }

    def _get_media_params(self, sig: Union[callable, inspect.Signature]) -> dict:
        """
        Identify media-related parameters in a function.

        Args:
            sig: Signature to analyze

        Returns:
            dict: Dictionary mapping parameter names to their media-related type annotations
        """
        annotations = self._sig_to_annotations(sig)
        return {key: annot for key, annot in annotations.items() if self._is_media_param(annot)}

    def _read_upload_files(self, files: dict, media_types: dict, *args, **kwargs) -> dict:
        """
        Read upload files from the request and convert them to MediaFile objects.
        :param files: dictionary of files to process. The keys are the parameter names and the values are the files.
        :param media_types: to which type the files should be converted. The keys are the parameter names and the values are the types.
        :return: dictionary of processed files
        If you want custom file handling, you can override this method.
        """
        # also convert those that now include a file upload even if was not specified in the signature
        converted_files = {}
        for key, value in files.items():
            # ignore empty values (default arguments) will not be converted to MediaFile
            if MediaDict._is_empty_file(value):
                continue

            try:
                converted_files[key] = self._convert_param_to_media_file(value, media_types.get(key, MediaFile))
            except Exception as e:
                raise ValueError(f"Could not parse file {key}. Check if the file is correct. Error: {str(e)}")
        return converted_files

    def _handle_file_uploads(self, func: Callable) -> Callable:
        """
        Wrap a function to handle file uploads and conversions.

        Args:
            func: Original function to wrap

        Returns:
            Wrapped function with file conversion logic
        """
        sig = inspect.signature(func)
        media_params = self._get_media_params(sig)
     
        param_names = list(sig.parameters.keys())

        @functools.wraps(func)
        def file_upload_wrapper(*args, **kwargs):
            # Map positional arguments to parameter names
            named_args = {param_names[i]: arg for i, arg in enumerate(args) if i < len(param_names)}
            named_args.update(kwargs)

            # Convert media-related parameters
            files_to_process = {
                param_name: param_value
                for param_name, param_value in named_args.items()
                if param_name in media_params or MediaFile._is_starlette_upload_file(param_value)
            }

            try:
                processed_files = self._read_upload_files(files_to_process, media_params, *args, **kwargs)
            except Exception as e:
                raise FileUploadException(message=str(e))

            # Update arguments with converted files
            named_args.update(processed_files)
            return func(**named_args)

        return file_upload_wrapper
