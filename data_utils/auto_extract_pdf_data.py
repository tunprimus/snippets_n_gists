#!/usr/bin/env python3
import camelot
import cv2
import fitz
import numpy as np
import os
import pandas as pd
import pdfplumber
import pytesseract
import tabula
from os.path import realpath as realpath
from pdf2image import convert_from_path

# Step 1: Detect if PDF is Text-based or Scanned
def is_pdf_text_based(pdf_path):
    """
    Detect if a PDF is text-based or scanned (image-based).

    Given a PDF file path, detect if the PDF is text-based or scanned (image-based).
    Text-based PDFs are the ones that contain text that can be searched and copy-pasted.
    Scanned PDFs are the ones that contain images of text, and can only be searched using OCR.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file to detect.

    Returns
    -------
    bool
        True if the PDF is text-based, False if it is scanned (image-based).
    """
    real_path_to_pdf = realpath(pdf_path)
    doc = fitz.open(real_path_to_pdf)
    return any(page.get_text("text").strip() for page in doc)


# Step 2: Extract Text with PyMuPDF
def extract_text_with_pymupdf(pdf_path, output_dir):
    """
    Extract text from a text-based PDF using PyMuPDF and save it to a file.

    This function opens a PDF file, extracts text from each page using PyMuPDF,
    and writes the extracted text to a specified output directory.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file from which text will be extracted.
    output_dir : str
        The directory where the extracted text file will be saved.

    Outputs
    -------
    A file named 'extracted_text.txt' containing the extracted text from the PDF
    is saved in the specified output directory.

    Prints
    ------
    A success message indicating the location of the saved extracted text.
    """
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)
    doc = fitz.open(real_path_to_pdf)
    text = "\n".join([page.get_text("text") for page in doc])
    with open(os.path.join(output_dir, "extracted_text.txt"), "w", encoding="utf-8") as fop:
        fop.write(text)
    print(f"[SUCCESS] Extracted text with PyMuPDF to {real_path_to_output}.")



# Step 3: Extract Tables with Tabula
def extract_tables_with_tabula(pdf_path, output_dir):
    """
    Extract tables from a PDF file using Tabula and save them as CSV files.

    This function uses Tabula to extract tables from a PDF file and saves each
    extracted table as a separate CSV file in a specified output directory.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file from which tables will be extracted.
    output_dir : str
        The directory where the extracted tables will be saved as CSV files.

    Outputs
    -------
    One or more CSV files containing the extracted tables from the PDF are saved
    in the specified output directory.

    Returns
    -------
    bool
        True if the tables were successfully extracted and saved, False otherwise.

    Prints
    ------
    A success message indicating the location of the saved extracted tables if
    successful, otherwise an error message indicating the reason for failure.
    """
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)
    try:
        dfs = tabula.read_pdf(real_path_to_pdf, pages="all", multiple_tables=True)
        for i, df in enumerate(dfs):
            df.to_csv(os.path.join(output_dir, f"table_tabula_{i}.csv"), index=False)
        print(f"[SUCCESS] Extracted tables with Tabula to {real_path_to_output}.")
        return True
    except Exception as exc:
        print(f"[ERROR] Could not extract tables with Tabula: {exc}")
        return False


# Step 4: Extract Tables with pdfplumber if Tabula Fails
def extract_tables_with_pdfplumber(pdf_path, output_dir):
    """
    Extract tables from a PDF file using pdfplumber and save them as CSV files.

    This function uses pdfplumber to extract tables from a PDF file and saves each
    extracted table as a separate CSV file in a specified output directory.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file from which tables will be extracted.
    output_dir : str
        The directory where the extracted tables will be saved as CSV files.

    Outputs
    -------
    One or more CSV files containing the extracted tables from the PDF are saved
    in the specified output directory.

    Returns
    -------
    bool
        True if the tables were successfully extracted and saved, False otherwise.

    Prints
    ------
    A success message indicating the location of the saved extracted tables if
    successful, otherwise an error message indicating the reason for failure.
    """
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)
    try:
        with pdfplumber.open(real_path_to_pdf) as pdf:
            tables = []
            for page in pdf.pages:
                extracted_tables = page.extract_tables()
                for table in extracted_tables:
                    df = pd.DataFrame(table)
                    tables.append(df)
        for i, df in enumerate(tables):
            df.to_csv(os.path.join(output_dir, f"table_pdfplumber_{i}.csv"), index=False)
        print(f"[SUCCESS] Extracted tables with pdfplumber to {real_path_to_output}.")
        return True
    except Exception as exc:
        print(f"[ERROR] Could not extract tables with pdfplumber: {exc}")
        return False


# Step 5: Extract Tables with Camelot for Complex Layouts
def extract_tables_with_camelot(pdf_path, output_dir):
    """
    Extract tables from a PDF file using Camelot and save them as CSV files.

    This function utilizes Camelot to extract tables from a PDF file, specifically using the 'lattice' flavor,
    which is suitable for PDFs with tables that have visible lines between cells. It saves each extracted table
    as a separate CSV file in the specified output directory.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file from which tables will be extracted.
    output_dir : str
        The directory where the extracted tables will be saved as CSV files.

    Outputs
    -------
    One or more CSV files containing the extracted tables from the PDF are saved in the specified output directory.

    Returns
    -------
    bool
        True if the tables were successfully extracted and saved, False otherwise.

    Prints
    ------
    A success message indicating the location of the saved extracted tables if successful,
    otherwise an error message indicating the reason for failure.
    """
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)
    try:
        tables = camelot.read_pdf(real_path_to_pdf, pages="1", flavor="lattice")
        for i, table in enumerate(tables):
            table.df.to_csv(os.path.join(output_dir, f"table_camelot_{i}.csv"), index=False)
        print(f"[SUCCESS] Extracted tables with Camelot to {real_path_to_output}.")
        return True
    except Exception as exc:
        print(f"[ERROR] Could not extract tables with camelot: {exc}")
        return False


# Step 6: Extract Text from Scanned PDFs Using OCR
def extract_text_with_tesseract(pdf_path, output_dir):
    """
    Extract text from a PDF file using OCR (Optical Character Recognition) with Tesseract.

    This function uses Tesseract to extract text from a PDF file. It saves the extracted
    text to a file named "ocr_extracted_text.txt" in the specified output directory.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file from which text will be extracted.
    output_dir : str
        The directory where the extracted text will be saved.

    Outputs
    -------
    A file named 'ocr_extracted_text.txt' containing the extracted text from the PDF
    is saved in the specified output directory.

    Prints
    ------
    A success message indicating the location of the saved extracted text.
    """
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)
    pages = convert_from_path(real_path_to_pdf, dpi=300)
    text = "\n".join([pytesseract.image_to_string(page, config="--psm 6") for page in pages])
    with open(os.path.join(output_dir, "ocr_extracted_text.txt"), "w", encoding="utf-8") as fop:
        fop.write(text)
    print(f"[SUCCESS] Extracted text with OCR to {real_path_to_output}.")

# Step 7: Extract Images Using PyMuPDF
def extract_images_with_pymupdf(pdf_path, output_dir):
    """
    Extract images from a PDF file using PyMuPDF and save them as PNG files.

    This function opens a PDF file, extracts images from each page using PyMuPDF,
    and saves each extracted image as a separate PNG file in a specified output directory.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file from which images will be extracted.
    output_dir : str
        The directory where the extracted images will be saved as PNG files.

    Outputs
    -------
    One or more PNG files containing the extracted images from the PDF are saved
    in the specified output directory.

    Prints
    ------
    A success message indicating the location of the saved extracted images.
    """
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)
    doc = fitz.open(real_path_to_pdf)
    for page_num, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            with open(os.path.join(output_dir, f"image_{page_num}_{img_idx}.png"), "wb") as fop:
                fop.write(image_data)
    print(f"[SUCCESS] Extracted images from PDF to {real_path_to_output}.")


# Main Function to Automate Extraction
def auto_extract_pdf_data(pdf_path, output_dir="pdf_extracted_data"):
    """
    Automatically extract text, tables, and images from a PDF file.

    This function orchestrates the extraction of text, tables, and images from a given PDF file.
    It first determines if the PDF is text-based or scanned. For text-based PDFs, it extracts
    text using PyMuPDF and attempts to extract tables using Tabula, pdfplumber, or Camelot.
    For scanned PDFs, it uses OCR (Tesseract) for text extraction. Images are extracted
    using PyMuPDF regardless of the PDF type.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file to be processed.
    output_dir : str, optional
        The directory where the extracted data will be saved (default is 'pdf_extracted_data').

    Outputs
    -------
    Various files are saved in the specified output directory:
    - A text file containing the extracted text.
    - CSV files containing extracted tables, if any.
    - PNG files containing extracted images.

    Prints
    ------
    Informational messages indicating the extraction process and success messages
    for each type of data extracted.
    """

    os.makedirs(output_dir, exist_ok=True)
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)

    if is_pdf_text_based(real_path_to_pdf):
        print("[INFO] PDF contains selectable text. Extracting with PyMuPDF...")
        extract_text_with_pymupdf(real_path_to_pdf, output_dir)
        if not extract_tables_with_tabula(real_path_to_pdf, output_dir):
            if not extract_tables_with_pdfplumber(real_path_to_pdf, output_dir):
                extract_tables_with_camelot(real_path_to_pdf, output_dir)
    else:
        print("[INFO] PDF appears to be scanned. Using OCR...")
        extract_text_with_tesseract(real_path_to_pdf, output_dir)

    # Extract images
    extract_images_with_pymupdf(real_path_to_pdf, output_dir)


# Run the script
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 auto_extract_pdf_data.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    auto_extract_pdf_data(pdf_path)

