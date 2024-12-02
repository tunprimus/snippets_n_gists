# Adapted from https://gist.github.com/Trigary/cfff35c6a4db11a35c82d168a218bafb
# Usage
# Embed images inside sample-document.md and save output to embedded.md:
# python3 markdown-embed-img.py --input sample-document.md --output embedded.md
#
# Sample input
# Sample Document
# You can write anything here. Only references to local images will be modified.
#
# ![sample-image](./sample-image.png)
#
# Output
#
# ![sample-image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAmSURBVBhXY/wPBAxogAlKMzAyMkJZSIIgDTAJsCCIg6wSi5kMDACFOQsKKOVG2AAAAABJRU5ErkJggg==)
#
import argparse
import base64
import logging
import os
import re
import shutil
import tempfile
from typing import Optional, IO


IMAGE_PATTERN: re.Pattern = re.compile(r"!\[([^]]*)]\(([^)]+)\)")
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "svg", "bmp", "webp", "ico"}


def convert_source(source):
    """
    Converts an image file from the given source path to a base64-encoded data URI.

    Args:
        source (str): The path to the image file. If the path starts with "file://",
                      it will be removed before processing.

    Returns:
        str: A base64-encoded data URI string representing the image.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If there is an error reading the file or encoding it to base64.
    """
    if source.startswith("file://"):
        source = source[len("file://") :]
    with open(source, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
        return f"data:image/{os.path.splitext(source)[-1][1:]};base64,{encoded}"


def can_convert_source(source):
    """
    Determines if the provided image source can be converted to a base64-encoded data URI.

    Args:
        source (str): The path or URL of the image file.

    Returns:
        bool: True if the image source can be converted, False otherwise.

    The function performs the following checks:
    - Returns False if the source is already a base64-encoded data URI.
    - Returns False if the source is a non-file URL.
    - Logs a warning and returns False if the source has an unsupported extension.
    - Logs a warning and returns False if the source file does not exist in the filesystem.
    """
    logger = logging.getLogger(__name__)

    # Ignore already encoded images
    if source.startswith("data:"):
        return False

    # Ignore URLs (except file URLs)
    if source.startswith("http://") or source.startswith("https://"):
        return False

    # Log and skip unsupported extensions
    for extension in IMAGE_EXTENSIONS:
        if source.endswith(f".{extension}"):
            break
    else:
        logger.warning(f"Image source has unsupported extension: {source}")
        return False

    # Log and skip missing files
    if not os.path.exists(source):
        logger.warning(f"Image source not found in OS filesystem: {source}")
        return False

    return True


def convert_image(source, alt_text):
    """
    Converts an image file to a base64-encoded data URI and returns a markdown image tag.

    Args:
        source (str): The path or URL of the image file.
        alt_text (str): The alternate text for the image.

    Returns:
        str: A markdown image tag containing the base64-encoded data URI if conversion is successful;
             otherwise, a markdown image tag with the original source.

    Logs:
        Debug information about the conversion process and errors if the conversion fails.
    """
    logger = logging.getLogger(__name__)
    noop = f"![{alt_text}]({source})"

    if not can_convert_source(source):
        return noop

    try:
        logger.debug(f"Converting image: {source} to base64")
        return f"![{alt_text}]({convert_source(source)})"
    except Exception as exc:
        logger.error(f"Failed to convert image: {source} to base64", exc_info=exc)
        return noop


def convert_files(input_file: IO, output_file: IO):
    """
    Reads a markdown file and writes a new version of the file to the output file
    with all image tags converted to base64-encoded data URIs.

    Args:
        input_file (IO): An open file object for the input markdown file.
        output_file (IO): An open file object for the output markdown file.

    Logs:
        Debug information about the conversion process and errors if the conversion fails.
    """
    for line in input_file:
        for match in IMAGE_PATTERN.finditer(line):
            line = line.replace(match.group(0), convert_image(match.group(1), match.group(2)))
        output_file.write(line)


def convert_paths(input_path: str, output_path: Optional[str]) -> None:
    """
    Converts a markdown file by replacing image tags with base64-encoded data URIs and writes the result to the output path.

    If the output path is None, the input file is overwritten with the converted content.

    Args:
        input_path (str): The path to the input markdown file.
        output_path (str or None): The path to the output file. If None, the input file is overwritten.

    Raises:
        Exception: If an error occurs during file conversion or writing.
    """
    if output_path is None:
        with tempfile.NamedTemporaryFile("w") as output_temp:
            with open(input_path, "r") as input_file:
                convert_files(input_file, output_temp)
            shutil.copyfile(output_temp.name, input_path)
    else:
        with open(output_path, "w") as output_file:
            with open(input_path, "r") as input_file:
                convert_files(input_file, output_file)


def main():
    """
    Entry point for the script.

    Parses command line arguments and calls convert_path with the provided arguments.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output file")
    args = parser.parse_args()
    convert_paths(args.input, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
