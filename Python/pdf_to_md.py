#!/usr/bin/env python

def pdf_to_markdown(input_file, output_file="test-output.md"):
    import json
    import pathlib
    import pymupdf4llm
    import sys
    from os.path import realpath as realpath


    real_path_to_input_file = realpath(input_file)
    real_path_to_output_file = realpath(output_file)

    #md_text_with_tables = pymupdf4llm.to_markdown(doc=real_path_to_input_file,  page_chunks=True, write_images=True, image_path="./converted_pdf_to_markdown", image_format="png", dpi=300, extract_words=True)
    md_text_with_tables = pymupdf4llm.to_markdown(doc=real_path_to_input_file)
    #pathlib.Path(f"{output_file}").write_bytes(md_text_with_tables.encode())
    pathlib.Path(real_path_to_output_file).write_bytes(md_text_with_tables.encode())
    print("Markdown saved to output file")

def main(path_to_pdf, output_dir="./converted_pdf_to_markdown"):
    import os
    from os.path import realpath as realpath

    os.makedirs(output_dir, exist_ok=True)
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)
    pdf_to_markdown(real_path_to_pdf, real_path_to_output)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_md.py <path_to_pdf>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    main(pdf_path)
