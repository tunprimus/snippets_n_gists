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
    real_path_to_pdf = realpath(pdf_path)
    doc = fitz.open(real_path_to_pdf)
    return any(page.get_text("text").strip() for page in doc)


# Step 2: Extract Text with PyMuPDF
def extract_text_with_pymupdf(pdf_path, output_dir):
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)
    doc = fitz.open(real_path_to_pdf)
    text = "\n".join([page.get_text("text") for page in doc])
    with open(os.path.join(output_dir, "extracted_text.txt"), "w", encoding="utf-8") as fop:
        fop.write(text)
    print(f"[SUCCESS] Extracted text with PyMuPDF to {real_path_to_output}.")



# Step 3: Extract Tables with Tabula
def extract_tables_with_tabula(pdf_path, output_dir):
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
    real_path_to_pdf = realpath(pdf_path)
    real_path_to_output = realpath(output_dir)
    pages = convert_from_path(real_path_to_pdf, dpi=300)
    text = "\n".join([pytesseract.image_to_string(page, config="--psm 6") for page in pages])
    with open(os.path.join(output_dir, "ocr_extracted_text.txt"), "w", encoding="utf-8") as fop:
        fop.write(text)
    print(f"[SUCCESS] Extracted text with OCR to {real_path_to_output}.")

# Step 7: Extract Images Using PyMuPDF
def extract_images_with_pymupdf(pdf_path, output_dir):
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
def auto_extract_pdf_data(pdf_path, output_dir="extracted_data"):
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

