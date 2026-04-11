import inspect
from types import UnionType
from typing import Any, Union, get_args, get_origin, Callable, List, Dict
from apipod.engine.backend.fastapi.LimitedUploadFile import LimitedUploadFile
from apipod.engine.signatures.upload import is_param_media_toolkit_file
from apipod.engine.jobs.job_result import FileModel, ImageFileModel, AudioFileModel, VideoFileModel
from apipod.engine.signatures.policies import FastAPISignaturePolicies
from apipod.engine.files.base_file_mixin import _BaseFileHandlingMixin
from apipod.engine.utils import replace_func_signature
from media_toolkit import MediaList, MediaDict, ImageFile, AudioFile, VideoFile, MediaFile
import functools


class _fast_api_file_handling_mixin(_BaseFileHandlingMixin):
    """
    Handles file uploads and parameter conversions for APIPod.

    This mixin provides functionality to:
    1. Convert function parameters to request body parameters
    2. Handle file uploads from various sources (UploadFile, FileModel, Base64, URLs)
    3. Convert MediaFile responses to FileModel for API documentation
    4. Preserve original media type information in the OpenAPI schema via x-media-type metadata
    """
    def create_limited_upload_file(self, max_size_mb: float):
        """
        Factory function to create a subclass of LimitedUploadFile with a predefined max_size_mb.
        Needs to be done in factory function, because creating it directly causes pydantic errors
        """
        max_size_mb = max_size_mb if max_size_mb is not None else self.max_upload_file_size_mb

        class LimitedUploadFileWithMaxSize(LimitedUploadFile):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, max_size=max_size_mb, **kwargs)

        return LimitedUploadFileWithMaxSize

    def _get_file_model_annotation(self, arg: type, is_list: bool, max_upload_file_size_mb: float) -> Dict[str, Any]:
        """
        Extracts FileModel-like annotations from a type annotation.

        Args:
            annotation: Type annotation to extract FileModel from
        """
        _limited_upload_file = self.create_limited_upload_file(max_upload_file_size_mb)
        file_model_map = {
            ImageFile: ImageFileModel,
            AudioFile: AudioFileModel,
            VideoFile: VideoFileModel
        }

        file_model_annot = file_model_map.get(arg, FileModel)

        if is_list:
            return Union[List[_limited_upload_file], List[file_model_annot], List[str]]
        else:
            return Union[_limited_upload_file, file_model_annot, str]

    def _get_media_file_annotation(self, annotation: Any, max_upload_file_size_mb: float):
        """
        Converts MediaFile-like annotations into appropriate UploadFile types for FastAPI.
        Preserves original type information as metadata for OpenAPI schema.

        Args:
            annotation: Type annotation to convert
            max_upload_file_size_mb: Maximum file size in MB

        Returns:
            Tuple containing:
            - Converted type annotation suitable for FastAPI
            - Metadata dictionary with original type info (if applicable)
        """
        org_annotation = get_origin(annotation) or annotation

        # Handle Union/UnionType
        if org_annotation in [Union, UnionType]:
            args = get_args(annotation)

            # Check for MediaDict
            if any(arg == MediaDict for arg in args):
                raise ValueError("Use MediaList for declaring upload files instead of MediaDict")

            # Handle Union with MediaList
            if any(t == MediaList for t in args):
                non_media_params = [t for t in args if not self._is_media_param(t)]
                list_file_up_annot = self._get_file_model_annotation(MediaFile, is_list=True, max_upload_file_size_mb=max_upload_file_size_mb)
                #  First other types to give FastAPI the correct order (Users can enter values instead of uploading files)
                if not self._is_media_param(args[0]):
                    return Union[(*non_media_params, list_file_up_annot)]
                return Union[(list_file_up_annot, *non_media_params)]

            # Handle Union with MediaFile types
            if any(self._is_media_param(t) for t in args):
                non_media_params = [t for t in args if not self._is_media_param(t)]
                media_params = [t for t in args if self._is_media_param(t)]
                resolved_mps = []
                for mp in media_params:
                    rmp = self._get_file_model_annotation(mp, is_list=False, max_upload_file_size_mb=max_upload_file_size_mb)
                    resolved_mps.append(rmp)

                return Union[(*resolved_mps, *non_media_params)]

            return annotation

        # Handle MediaList
        if org_annotation == MediaList:
            generic_type = get_args(annotation)
            # Check for nested MediaList
            if any(t in (MediaList, MediaDict) for t in generic_type):
                raise ValueError("Nesting of MediaList/MediaDict is not supported")

            if len(generic_type) == 1 and self._is_media_param(generic_type[0]):
                return self._get_file_model_annotation(generic_type[0], is_list=True, max_upload_file_size_mb=max_upload_file_size_mb)

            return self._get_file_model_annotation(MediaFile, is_list=True, max_upload_file_size_mb=max_upload_file_size_mb)

        # Handle List types
        if org_annotation in [List, list]:
            args = get_args(annotation)
            list_file_up_annot = self._get_file_model_annotation(
                MediaFile,
                is_list=True,
                max_upload_file_size_mb=max_upload_file_size_mb
            )

            # Check for MediaDict
            if any(t == MediaDict for t in args):
                raise ValueError("Use MediaList for declaring upload files instead of MediaDict.")

            # Case List[List[MediaFile]] and List[MediaList] -> not allowed
            media_params = [t for t in args if self._is_media_param(t)]
            if len(media_params) == 0:
                return annotation

            if any(get_origin(t) in [List, list, MediaList] for t in args):
                raise ValueError("Nesting of MediaList and List is not supported")

            non_media_params = [t for t in args if t not in media_params]
            #  First other types to give FastAPI the correct order (Users can enter values instead of uploading files)
            if not self._is_media_param(args[0]):
                return Union[(*non_media_params, list_file_up_annot)]

            if len(media_params) == 1:
                return Union[(self._get_file_model_annotation(media_params[0], is_list=True, max_upload_file_size_mb=max_upload_file_size_mb), *non_media_params)]

            return Union[(list_file_up_annot, *non_media_params)]

        # Handle direct MediaFile types
        if is_param_media_toolkit_file(annotation):
            return self._get_file_model_annotation(annotation, is_list=False, max_upload_file_size_mb=max_upload_file_size_mb)

        # Handle FileModel
        if inspect.isclass(annotation) and issubclass(annotation, FileModel):
            return annotation

        return annotation

    def _convert_params_to_body(self, func: Callable, max_upload_file_size_mb: float = None) -> List[inspect.Parameter]:
        """
        Moves all parameters to the request body.
        Replaces MediaFile parameters with UploadFile in the function signature.
        Preserves original type information as metadata for OpenAPI schema.
        This allows the API to accept file uploads from the client while documenting
        the expected file type correctly.
        """
        sig = inspect.signature(func)
        annotations = self._sig_to_annotations(sig)

        fastapi_dependencies_parameters = []
        field_definitions = {}
        for name, param in sig.parameters.items():
            # Skip FastAPI dependency injections like Depends, Security, Body, Request, Response
            if FastAPISignaturePolicies.is_fastapi_dependency(param):
                fastapi_dependencies_parameters.append(param)
                continue
            annotation = annotations.get(name, Any)
            default = param.default if param.default != inspect.Parameter.empty else ...

            # Check if the parameter was originally Optional
            is_optional = get_origin(annotation) in {Union, UnionType} and type(None) in get_args(annotation)

            # Convert and check if was converted
            _file_annotation = self._get_media_file_annotation(annotation, max_upload_file_size_mb)
            is_file_parameter = annotation != _file_annotation
            annotation = _file_annotation

            # Move to body parameters with appropriate metadata
            if is_file_parameter and is_optional:
                file_args = get_args(_file_annotation)
                # adding str, and None to the union to allow empty strings, and none values
                annotation = Union[(*file_args, None)]
                default = default if default is not ... else None

            if not is_file_parameter:
                default = FastAPISignaturePolicies.build_non_file_default(
                    annotation=annotation,
                    default=default,
                    is_optional=is_optional
                )

            field_definitions[name] = (annotation, default)

        parameters = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=param_type, default=default)
            for name, (param_type, default) in field_definitions.items()
        ]
        parameters.extend(fastapi_dependencies_parameters)

        return parameters

    def _update_signature(self, func: Callable, max_upload_file_size_mb: float = None) -> Callable:
        """
        Update the function signature with the converted parameter definitions.
        
        Args:
            func: Function to update
            max_upload_file_size_mb: Maximum file size in MB
            
        Returns:
            Function with updated signature
        """
        body_params = self._convert_params_to_body(func, max_upload_file_size_mb)
        func = replace_func_signature(func, inspect.Signature(parameters=body_params))
        return func

    def _prepare_func_for_media_file_upload_with_fastapi(self, func: callable, max_upload_file_size_mb: float = None) -> callable:
        """
        Prepare a function for FastAPI
        1. Adds file upload logic to convert parameters
        2. Injects a dummy JobProgress if the function expects it but we're not using a queue
        3. Removes job progress parameter from the function signature (for OpenAPI docs)
        4. Replaces upload file parameters with FastAPI File type
        """
        # 1. Add file upload conversion logic (must be first to handle positional args)
        file_upload_modified = self._handle_file_uploads(func)

        # 2. Inject dummy JobProgress if needed (for non-queued endpoints)
        with_dummy_progress = self._inject_dummy_job_progress(file_upload_modified)

        # 3. Remove job progress parameter from signature (so it doesn't show in OpenAPI)
        no_job_progress = self._remove_job_progress_from_signature(with_dummy_progress)

        # 4. Update signature with file upload parameters (converts to Form/File)
        with_file_upload_signature = self._update_signature(no_job_progress, max_upload_file_size_mb)

        return with_file_upload_signature

    def _inject_dummy_job_progress(self, func: Callable) -> Callable:
        """
        Wrap the function to inject a dummy JobProgress instance if the original
        function expects one. This is used for non-queued FastAPI endpoints.
        """
        from apipod.engine.jobs.job_progress import JobProgress
        
        sig = inspect.signature(func)
        job_progress_params = [
            p.name for p in sig.parameters.values()
            if p.name == "job_progress" or "FastJobProgress" in str(p.annotation)
        ]

        if not job_progress_params:
            return func

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                for param_name in job_progress_params:
                    if param_name not in kwargs:
                        kwargs[param_name] = JobProgress()
                return await func(*args, **kwargs)
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for param_name in job_progress_params:
                    if param_name not in kwargs:
                        kwargs[param_name] = JobProgress()
                return func(*args, **kwargs)

        return wrapper

    def _remove_job_progress_from_signature(self, func: Callable) -> Callable:
        """
        Remove job_progress parameter from function signature for API docs.

        Args:
            func: Function to modify

        Returns:
            Function with updated signature
        """
        sig = inspect.signature(func)
        new_sig = sig.replace(parameters=[
            p for p in sig.parameters.values()
            if p.name != "job_progress" and "FastJobProgress" not in str(p.annotation)
        ])
        if len(new_sig.parameters) != len(sig.parameters):
            return replace_func_signature(func, new_sig)

        return func
