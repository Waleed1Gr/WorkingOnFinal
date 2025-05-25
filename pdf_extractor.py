import PyPDF2
import os
import re

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk.strip())
        
        # Break if we've reached the end
        if i + chunk_size >= len(words):
            break
    
    return chunks

def save_chunks_to_file(chunks, output_file="data.txt"):
    """Save chunks to text file, each chunk on separate line"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk + '\n')
        print(f"Successfully saved {len(chunks)} chunks to {output_file}")
    except Exception as e:
        print(f"Error saving to file: {e}")

def process_pdf(pdf_path, output_file="data.txt", chunk_size=500, overlap=50):
    """Main function to process PDF and save chunks"""
    print(f"Processing PDF: {pdf_path}")
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return False
    
    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        print("Failed to extract text from PDF")
        return False
    
    # Clean text
    cleaned_text = clean_text(raw_text)
    print(f"Extracted {len(cleaned_text)} characters from PDF")
    
    # Create chunks
    chunks = chunk_text(cleaned_text, chunk_size, overlap)
    print(f"Created {len(chunks)} chunks")
    
    # Save to file
    save_chunks_to_file(chunks, output_file)
    
    return True

if __name__ == "__main__":
    # Example usage
    pdf_file = input("Enter PDF file path: ").strip()
    
    if pdf_file:
        success = process_pdf(pdf_file)
        if success:
            print("PDF processing completed successfully!")
            print("You can now run 'python local_rag.py' to start querying your document.")
        else:
            print("PDF processing failed.")
    else:
        print("No PDF file specified.")