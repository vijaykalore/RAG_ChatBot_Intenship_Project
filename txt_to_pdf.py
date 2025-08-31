import os
from fpdf import FPDF

IN_DIR = "texts"
OUT_DIR = "pdfs"
os.makedirs(OUT_DIR, exist_ok=True)

for fname in os.listdir(IN_DIR):
    if not fname.endswith(".txt"):
        continue
    in_path = os.path.join(IN_DIR, fname)
    out_path = os.path.join(OUT_DIR, os.path.splitext(fname)[0] + ".pdf")
    with open(in_path, "r", encoding="utf-8") as f:
        text = f.read()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 6, line)
    pdf.output(out_path)

print(f"Converted text files in {IN_DIR}/ to PDFs in {OUT_DIR}/")
