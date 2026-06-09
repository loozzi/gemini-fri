class SDKError(Exception):
    pass


class APIError(SDKError):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")


class AuthError(APIError):
    """401 Unauthorized"""


class RateLimitError(APIError):
    """429 Too Many Requests"""


class ServerError(APIError):
    """5xx Server Error"""


def raise_for_status(status_code: int, message: str) -> None:
    if status_code == 401:
        raise AuthError(status_code, message)
    if status_code == 429:
        raise RateLimitError(status_code, message)
    if status_code >= 500:
        raise ServerError(status_code, message)
    if status_code >= 400:
        raise APIError(status_code, message)
