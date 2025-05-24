#!/usr/bin/env python3
"""
PDF Reader Tool for Im2Height Project

This script extracts text content from PDF files, particularly useful for
reading research papers and documentation. It can extract text from the
Im2Height paper or any other PDF documents.

Usage:
    python tools/read_pdf.py <pdf_file_path>
    python tools/read_pdf.py documents/IM2HEIGHT*.pdf

Features:
    - Extract text from PDF files
    - Handle multiple pages
    - Clean text formatting
    - Save extracted text to file
    - Support for research paper analysis
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Installing...")
    os.system("pip install PyPDF2")
    import PyPDF2

try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except ImportError:
    print("pdfplumber not found. Will use PyPDF2 only.")
    PDF_PLUMBER_AVAILABLE = False


def extract_text_pypdf2(pdf_path):
    """Extract text using PyPDF2 library."""
    text_content = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"PDF has {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                print(f"Extracting page {page_num}...")
                text = page.extract_text()
                if text.strip():
                    text_content.append(f"\n--- PAGE {page_num} ---\n")
                    text_content.append(text)
                    
    except Exception as e:
        print(f"Error extracting with PyPDF2: {e}")
        return None
        
    return '\n'.join(text_content)


def extract_text_pdfplumber(pdf_path):
    """Extract text using pdfplumber library (better formatting)."""
    text_content = []
    
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            print(f"PDF has {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"Extracting page {page_num}...")
                text = page.extract_text()
                if text and text.strip():
                    text_content.append(f"\n--- PAGE {page_num} ---\n")
                    text_content.append(text)
                    
    except Exception as e:
        print(f"Error extracting with pdfplumber: {e}")
        return None
        
    return '\n'.join(text_content)


def clean_text(text):
    """Clean and format extracted text."""
    if not text:
        return ""
        
    # Basic text cleaning
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Remove excessive whitespace
            line = ' '.join(line.split())
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def save_text_to_file(text, pdf_path, output_dir=None):
    """Save extracted text to a .txt file."""
    if not output_dir:
        output_dir = os.path.dirname(pdf_path)
    
    # Create output filename
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{pdf_name}.txt")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving text: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract text from PDF files')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--output', '-o', help='Output directory for text file')
    parser.add_argument('--method', choices=['pypdf2', 'pdfplumber', 'auto'], 
                        default='auto', help='Extraction method to use')
    parser.add_argument('--save', action='store_true', 
                        help='Save extracted text to file')
    parser.add_argument('--clean', action='store_true', 
                        help='Clean text formatting')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: File '{args.pdf_path}' not found!")
        sys.exit(1)
    
    print(f"Extracting text from: {args.pdf_path}")
    
    # Choose extraction method
    text = None
    if args.method == 'auto':
        # Try pdfplumber first (better quality), fall back to PyPDF2
        if PDF_PLUMBER_AVAILABLE:
            print("Using pdfplumber for extraction...")
            text = extract_text_pdfplumber(args.pdf_path)
        
        if not text:
            print("Falling back to PyPDF2...")
            text = extract_text_pypdf2(args.pdf_path)
    
    elif args.method == 'pdfplumber':
        if PDF_PLUMBER_AVAILABLE:
            text = extract_text_pdfplumber(args.pdf_path)
        else:
            print("pdfplumber not available, using PyPDF2")
            text = extract_text_pypdf2(args.pdf_path)
    
    elif args.method == 'pypdf2':
        text = extract_text_pypdf2(args.pdf_path)
    
    if not text:
        print("Failed to extract text from PDF!")
        sys.exit(1)
    
    # Clean text if requested
    if args.clean:
        print("Cleaning text...")
        text = clean_text(text)
    
    # Save to file if requested
    if args.save:
        save_text_to_file(text, args.pdf_path, args.output)
    else:
        # Print to stdout
        print("\n" + "="*50)
        print("EXTRACTED TEXT:")
        print("="*50)
        print(text)
    
    print(f"\nExtraction completed! Text length: {len(text)} characters")


if __name__ == "__main__":
    main()
