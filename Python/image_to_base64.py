import base64
import sys

def image_to_base64(image_file, output_file):
    """
    Takes an image file and writes a file with the image encoded in base64 format for use in HTML.

    Args:
        image_file (str): The path to the image file to be read.
        output_file (str): The path to where the base64 encoded image should be written.
    """
    with open(image_file, "rb") as image:
        image_read = image.read()
        image_64_encode = base64.encodestring(image_read) if sys.version_info <(3,9) else base64.encodebytes(image_read)
        image_string = str(image_64_encode)
        image_string = image_string.replace("\\n", "")
        image_string = image_string.replace("b'", "")
        image_string = image_string.replace("'", "")
        image_string = '<p><img src="data:image/png;base64,' + image_string + '"></p>'
        image_result = open(output_file, "w")
        image_result.write(image_string)
