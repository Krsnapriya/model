import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Pediatric Disease Prediction - Codebase Reference', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, f'{label}', 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                txt = f.read()
            
            # Sanitize for latin-1
            replacements = {
                '\u2014': '--',
                '\u2013': '-',
                '\u2018': "'",
                '\u2019': "'",
                '\u201c': '"',
                '\u201d': '"',
                '\u2022': '*',
            }
            for char, replacement in replacements.items():
                txt = txt.replace(char, replacement)
            
            # Final safety net: replace any remaining non-latin-1 characters with ?
            txt = txt.encode('latin-1', 'replace').decode('latin-1')

        except Exception as e:
            txt = f"Error reading file: {str(e)}"

        self.set_font('Courier', '', 10) # Monospace for code
        
        # Split text into lines to handle images
        lines = txt.split('\n')
        for line in lines:
            if line.strip().startswith('[IMAGE:') and line.strip().endswith(']'):
                # Extract image path
                img_path = line.strip()[7:-1].strip()
                # Check for absolute path mapping if needed, or assume local
                if os.path.exists(img_path):
                    self.ln(5)
                    try:
                        # Center the image
                        x = self.get_x()
                        self.image(img_path, w=170) 
                    except Exception as img_err:
                        self.cell(0, 5, f"[Error loading image: {img_err}]", 0, 1)
                    self.ln(5)
                else:
                     self.cell(0, 5, f"[Image not found: {img_path}]", 0, 1)
            else:
                self.multi_cell(0, 5, line)
        self.ln()

    def print_file(self, filepath):
        self.add_page()
        self.chapter_title(filepath)
        self.chapter_body(filepath)

def generate_pdf():
    pdf = PDF()
    pdf.alias_nb_pages()
    
    # List of files to include
    files_to_print = [
        "CODEBASE_EXPLANATION.md",
        "REPORT.md",
        "requirements.txt",
        "docker/Dockerfile",
        "train_model.py",
        "dashboard.py",
        "api/app.py",
        "data/synthetic_data_generator.py",
        "create_notebook_helper.py",
        "train_model_fallback.py"
    ]

    for filepath in files_to_print:
        if os.path.exists(filepath):
            print(f"Adding {filepath}...")
            pdf.print_file(filepath)
        else:
            print(f"Skipping {filepath} (not found)")

    output_path = "Pediatric_Disease_Prediction_Codebase.pdf"
    pdf.output(output_path, 'F')
    print(f"\nPDF generated successfully: {output_path}")

if __name__ == "__main__":
    generate_pdf()
