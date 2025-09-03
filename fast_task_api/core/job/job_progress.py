class JobProgress:
    def __init__(self, progress: float = 0, message: str = None):
        """
        Used to display _progress of a job while executing.
        :param progress: value between 0 and 1.0
        :param message: message to deliver to client.
        """
        self._progress = progress
        self._message = message

    def set_status(self, progress: float = None, message: str = None):
        if progress is not None:
            self._progress = progress
        if message is not None:
            self._message = message


class JobProgressRunpod(JobProgress):
    def __init__(self, runpod_job, progress: float = 0, message: str = None):
        super().__init__(progress=progress, message=message)
        self.runpod_job = runpod_job

    def set_status(self, progress: float = None, message: str = None):
        super().set_status(progress=progress, message=message)

        try:
            import runpod
            runpod.serverless.progress_update(
                self.runpod_job,
                f"Progress: {int(self._progress)} Message: {self._message}"
            )
        except Exception as e:
            print(f"Problem in progress update: {e}")
