import logging
import sys


def setup_logging(log_level):
    get_logger().setLevel(log_level)

    log_output_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s - %(message)s [%(filename)s:%(lineno)s] [%(relativeCreated)d]')

    stdout_stream_handler = logging.StreamHandler(sys.stdout)
    stdout_stream_handler.setLevel(log_level)
    stdout_stream_handler.setFormatter(log_output_formatter)

    get_logger().addHandler(stdout_stream_handler)

    stderr_stream_handler = logging.StreamHandler(sys.stdout)
    stderr_stream_handler.setLevel(logging.WARNING)
    stderr_stream_handler.setFormatter(log_output_formatter)

    get_logger().addHandler(stderr_stream_handler)


def get_logger() -> logging.Logger:
    return logging.getLogger('lane-detector')
