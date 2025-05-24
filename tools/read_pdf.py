#!/usr/bin/env python
import os
import sys
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 is not installed. Please run: mamba install -y pypdf2")
    sys.exit(1)

def read_pdf(pdf_path):
    """
    Read and extract text from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return None
        
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Number of pages in the PDF: {num_pages}")
            
            # Extract text from each page
            full_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                full_text += f"\n--- Page {page_num + 1} ---\n"
                full_text += page.extract_text()
                
            return full_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

if __name__ == "__main__":
    pdf_path = "documents/IM2HEIGHT - Height Estimation from Single Monocular Imagery via Fully Residual Convolutional-Deconvolutional Network.pdf"
    
    # Get absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_absolute_path = os.path.join(current_dir, pdf_path)
    
    print(f"Reading PDF file: {pdf_absolute_path}")
    text = read_pdf(pdf_absolute_path)
    
    if text:
        # Print first 1000 characters to avoid overwhelming output
        print("\nPDF Content (first 1000 characters):")
        print(text[:1000])
        print("\n...")
        
        # Optionally save to a text file
        txt_path = pdf_absolute_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        print(f"\nFull text saved to: {txt_path}")