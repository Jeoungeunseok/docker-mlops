class AppError(Exception):
    """Base application exception."""


class ModelNotLoadedError(AppError):
    pass


class ModelPromotionError(AppError):
    pass
