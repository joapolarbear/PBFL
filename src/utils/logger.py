import logging
import os, sys
import time

LOG_LEVEL_NAME = ["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"]

## Ref: https://stackoverflow.com/questions/12980512/custom-logger-class-and-correct-line-number-function-name-in-log
# This code is mainly copied from the python logging module, with minor modifications

# _srcfile is used when walking the stack to check when we've got the first
# caller stack frame.
#
if hasattr(sys, 'frozen'): #support for py2exe
    _srcfile = "logging%s__init__%s" % (os.sep, __file__[-4:])
elif __file__[-4:].lower() in ['.pyc', '.pyo']:
    _srcfile = __file__[:-4] + '.py'
else:
    _srcfile = __file__
_srcfile = os.path.normcase(_srcfile)

class RelativeSeconds(logging.Formatter):
    def format(self, record):
        # record.relativeCreated = record.relativeCreated // 1000
        record.delta = record.relativeCreated / 1000
        return super().format(record)
    
class Logger:
    def __init__(self, args, path, name, logging_level="INFO", is_clean=False, show_progress=False):
        exp_name = os.environ.get("PBFL_EXP_NAME", None)
        if exp_name is not None:
            logfile = exp_name + "-log.txt"
        else:
            dirname = path if os.path.isdir(path) else os.path.dirname(path)
            dirname = os.path.join(path, ".log")
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            logfile = os.path.join(dirname, f"log_option-{args.method.lower()}.txt")
        if is_clean and os.path.exists(logfile):
            os.remove(logfile)
        #! config logging
        self.logger = logging.getLogger(name)
        log_level = logging_level.lower()
        if log_level == "trace":
            _log_level = logging.TRACE
        elif log_level == "debug":
            _log_level = logging.DEBUG
        elif log_level == "warn" or log_level == "warning":
            _log_level = logging.WARNING
        elif log_level == "error":
            _log_level = logging.ERROR
        else:
            _log_level = logging.INFO
        self.logger.setLevel(level=_log_level)

        formatter = RelativeSeconds('[%(asctime)s(+%(delta)ss)] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

        #! bind some file stream
        handler = logging.FileHandler(logfile)
        handler.setLevel(_log_level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if not show_progress: 
            #! if we want show progress, no need to bind the output stream 
            console = logging.StreamHandler()
            console.setLevel(_log_level)
            console.setFormatter(formatter)
            self.logger.addHandler(console)

    def info(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log(logging.ERROR, msg, args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log(logging.DEBUG, msg, args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, args, **kwargs)

    def _log(self, level, msg, args, exc_info=None, extra=None):
        """
        Low-level logging routine which creates a LogRecord and then calls
        all the handlers of this logger to handle the record.
        """
        # Add wrapping functionality here.
        if _srcfile:
            #IronPython doesn't track Python frames, so findCaller throws an
            #exception on some versions of IronPython. We trap it here so that
            #IronPython can use logging.
            try:
                fn, lno, func = self.findCaller()
            except ValueError:
                fn, lno, func = "(unknown file)", 0, "(unknown function)"
        else:
            fn, lno, func = "(unknown file)", 0, "(unknown function)"
        if exc_info:
            if not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()
        record = self.logger.makeRecord(
            self.logger.name, level, fn, lno, msg, args, exc_info, func, extra)
        self.logger.handle(record)


    def findCaller(self):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = logging.currentframe()
        #On some versions of IronPython, currentframe() returns None if
        #IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        rv = "(unknown file)", 0, "(unknown function)"
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _srcfile:
                f = f.f_back
                continue
            rv = (co.co_filename, f.f_lineno, co.co_name)
            break
        return rv