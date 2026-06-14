import functools
import inspect
from types import UnionType
from typing import Any, Optional, Union, get_args, get_origin, Callable, List, Type

from pydantic import BaseModel

from media_toolkit import media_from_any, MediaFile, MediaList, MediaDict, ImageFile, AudioFile, VideoFile
from apipod.engine.signatures.upload import is_param_media_toolkit_file
from apipod.common.schemas.media_files import FileModel, ImageFileModel, AudioFileModel, VideoFileModel
from apipod.common.exceptions import FileUploadException


# Wire-format pydantic file models mapped to the media-toolkit type they parse into.
_FILE_MODEL_MEDIA_TYPES: dict = {
    ImageFileModel: ImageFile,
    AudioFileModel: AudioFile,
    VideoFileModel: VideoFile,
    FileModel: MediaFile,
}


def _media_type_for_file_model(file_model_cls: Type) -> Type:
    """Resolve the media-toolkit type a FileModel subclass should be parsed into."""
    for cls in file_model_cls.__mro__:
        if cls in _FILE_MODEL_MEDIA_TYPES:
            return _FILE_MODEL_MEDIA_TYPES[cls]
    return MediaFile


def _parse_file_model_value(value: Any) -> Any:
    """Convert FileModel instances (also inside lists) into parsed media-toolkit objects."""
    if isinstance(value, FileModel):
        return media_from_any(
            data=value.model_dump(include={"file_name", "content_type", "content"}),
            type_hint=_media_type_for_file_model(type(value)),
            use_temp_file=True,
            temp_dir=None,
            allow_reads_from_disk=False,
        )
    if isinstance(value, list):
        return [_parse_file_model_value(item) for item in value]
    return value


def parse_schema_media_fields(schema_obj):
    """
    Replace FileModel-typed fields of a validated request schema with parsed
    media-toolkit objects, so endpoint functions receive ready-to-use media —
    exactly as method-level ``def endpoint(image: ImageFile)`` parameters do.

    Lives in the file-handling layer because it is the schema-side counterpart
    of the parameter-level upload conversion (``_convert_param_to_media_file``).
    """
    for field_name in type(schema_obj).model_fields:
        value = getattr(schema_obj, field_name, None)
        parsed = _parse_file_model_value(value)
        if parsed is not value:
            # The runtime type (e.g. ImageFile) intentionally differs from the wire
            # annotation (ImageFileModel); bypass pydantic's assignment validation.
            object.__setattr__(schema_obj, field_name, parsed)
    return schema_obj


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

            if target_type == MediaList:
                return MediaList(
                    read_system_files=False,
                    download_files=True,
                    use_temp_file=True,
                    temp_dir=None
                ).from_any(param_value)

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

    def _resolve_schema_annotation(self, annotation: Any) -> Optional[Type]:
        """
        Resolve an annotation (also ``Optional``/``Union``) to a pydantic request
        schema whose nested FileModel fields should be parsed.

        FileModel variants are themselves media handled by the parameter-level upload
        path (``_convert_param_to_media_file``), so they are excluded here.
        """
        candidates = [annotation]
        if get_origin(annotation) in (Union, UnionType):
            candidates = [arg for arg in get_args(annotation) if arg is not type(None)]

        for candidate in candidates:
            if inspect.isclass(candidate) and issubclass(candidate, BaseModel) and not issubclass(candidate, FileModel):
                return candidate
        return None

    def _parse_schema_params(self, schema_params: set, named_args: dict) -> dict:
        """
        Parse nested FileModel fields inside request-schema arguments.

        Called by ``_handle_file_uploads`` before the media-param conversion so that
        endpoint functions always receive ready-to-use media-toolkit objects regardless
        of whether files arrive as top-level parameters or nested inside a schema.

        Args:
            schema_params: Parameter names whose annotation resolved to a pydantic schema.
            named_args: Current mapping of parameter name → value for this call.

        Returns:
            Mapping of parameter name → schema instance with media fields replaced.
        """
        parsed = {}
        for name in schema_params:
            value = named_args.get(name)
            if value is None:
                continue
            try:
                parsed[name] = parse_schema_media_fields(value)
            except Exception as e:
                raise FileUploadException(message=f"Could not parse media fields of '{name}': {e}")
        return parsed

    def _handle_file_uploads(self, func: Callable) -> Callable:
        """
        Wrap a function to handle file uploads and conversions.
        Handles both synchronous and asynchronous functions correctly.

        Args:
            func: Original function to wrap

        Returns:
            Wrapped function with file conversion logic
        """
        sig = inspect.signature(func)
        media_params = self._get_media_params(sig)

        # Parameters typed as a pydantic request schema whose nested FileModel fields
        # must be parsed into media-toolkit objects before the endpoint executes.
        schema_params = {
            name for name, annot in self._sig_to_annotations(sig).items()
            if name not in media_params and self._resolve_schema_annotation(annot) is not None
        }

        param_names = list(sig.parameters.keys())

        # Check if the original function is async
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def file_upload_wrapper(*args, **kwargs):
                # Map positional arguments to parameter names
                named_args = {param_names[i]: arg for i, arg in enumerate(args) if i < len(param_names)}
                named_args.update(kwargs)

                # Parse nested media inside schema parameters
                named_args.update(self._parse_schema_params(schema_params, named_args))

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

                # AWAIT the async function
                return await func(**named_args)
        else:
            @functools.wraps(func)
            def file_upload_wrapper(*args, **kwargs):
                # Map positional arguments to parameter names
                named_args = {param_names[i]: arg for i, arg in enumerate(args) if i < len(param_names)}
                named_args.update(kwargs)

                # Parse nested media inside schema parameters
                named_args.update(self._parse_schema_params(schema_params, named_args))

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

                # Call the sync function directly
                return func(**named_args)

        return file_upload_wrapper