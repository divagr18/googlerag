import fitz  # PyMuPDF
import tiktoken

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def main():
    pdf_path = r"C:\Users\Keshav\Downloads\policy (1).pdf"
    text = extract_text_from_pdf(pdf_path)
    token_count = count_tokens(text)
    
    print(f"\nTotal tokens in '{pdf_path}': {token_count}")

if __name__ == "__main__":
    main()
