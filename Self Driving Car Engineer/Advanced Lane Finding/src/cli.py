import logging

import click

from src.detector import Detector
from src.logger import setup_logging


@click.group("lane-detector")
@click.option('--debug', default=False, help='Set the logs to debug level', show_default=True, is_flag=True)
def main(debug):
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)


@click.command('detect-video', help='Starts lane detection on video')
@click.option("--video_path", envvar='VIDEO_PATH')
def detect_video(video_path: str) -> None:
    detector = Detector()
    detector.start(video_path)


@click.command('detect-images', help='Starts lane detection on images')
@click.option("--images_directory", envvar='IMAGES_DIRECTORY')
@click.option("--calibration_directory", envvar='CALIBRATION_DIRECTORY')
@click.option("--output_directory", envvar='OUTPUT_DIRECTORY')
def detect_images(images_directory: str, calibration_directory: str, output_directory: str) -> None:
    detector = Detector()
    detector.build(images_directory, calibration_directory, output_directory)
    detector.start()


main.add_command(detect_video)
main.add_command(detect_images)

if __name__ == "__main__":
    main()
