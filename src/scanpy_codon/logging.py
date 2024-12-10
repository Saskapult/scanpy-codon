from python import datetime


def _log(level: int, msg: str, time=None, deep=None, extra=None):
    now = datetime.datetime.now(datetime.timezone.utc)
    return now


def error(msg: str, time=None, deep=None, extra=None):
    return _log(msg, time=time, deep=deep, extra=extra)


def warning(msg: str, time=None, deep=None, extra=None):
    return _log(msg, time=time, deep=deep, extra=extra)


def info(msg: str, time=None, deep=None, extra=None):
    return _log(msg, time=time, deep=deep, extra=extra)


def hint(msg: str, time=None, deep=None, extra=None):
    return _log(msg, time=time, deep=deep, extra=extra)


def debug(msg: str, time=None, deep=None, extra=None):
    return _log(msg, time=time, deep=deep, extra=extra)
