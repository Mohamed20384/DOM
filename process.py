import json
import os
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx2pdf import convert

# === Settings ===
INPUT_FILE = "Restaurants.json"
DOCX_DIR = "Restaurants_Word"
PDF_DIR = "Restaurants_PDF"

# === Create output directories ===
os.makedirs(DOCX_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# === Generate Word Docs ===
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

for rest in data:
    name = rest.get("name", "مطعم بدون اسم")
    address = rest.get("adress", "")
    phone = rest.get("phone", [])
    rank = rest.get("rank", "غير مصنف")
    bestsell = rest.get("bestsell", "غير محدد")  # ✅ Added bestsell as تصنيف المطعم
    real_rates = rest.get("realRates", {})
    emenu = rest.get("eMenu", {})
    open_time = rest.get("openTime", "غير معروف")
    close_time = rest.get("closeTime", "غير معروف")
    cooking_time = rest.get("cookingTimeRange", "غير محدد")

    doc = Document()

    def write_paragraph(label, value):
        if isinstance(value, list):
            value = "، ".join(value)
        elif isinstance(value, dict):
            value = "\n".join([f"{k}: {v}" for k, v in value.items()])
        text = f"{label}: {value}"
        para = doc.add_paragraph(text)
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = para.runs[0]
        run.font.name = 'Calibri'
        run.font.size = Pt(14)

    # === Content ===
    write_paragraph("اسم المطعم", name)
    write_paragraph("العنوان", address)
    write_paragraph("أرقام الهاتف", phone)
    write_paragraph("ترتيب المطعم", rank)      
    write_paragraph("تصنيف المطعم", bestsell)   

    # ✅ Ratings section without emails
    write_paragraph("التقييمات", "")
    for info in real_rates.values():
        rate = info.get("rate", "")
        comment = info.get("comment", "")
        rate_text = f"التقييم: {rate}"
        comment_text = f"التعليق: {comment}" if comment else ""
        write_paragraph("", f"{rate_text}\n{comment_text}")

    write_paragraph("القائمة الإلكترونية", "")
    for section_name, section in emenu.items():
        write_paragraph("القسم", section_name)
        if 'products' in section:
            for product in section['products']:
                write_paragraph(" - الاسم", product.get("name", ""))
                write_paragraph("   الوصف", product.get("desc", ""))
                if 'price' in product:
                    write_paragraph("   السعر", product['price'])
                if 'sizes' in product:
                    for size, price in product['sizes'].items():
                        write_paragraph(f"   الحجم: {size}", price)
                if 'extras' in product:
                    for extra, extra_price in product['extras'].items():
                        write_paragraph(f"   إضافة: {extra}", extra_price)

    write_paragraph("وقت الفتح", str(open_time))
    write_paragraph("وقت الإغلاق", str(close_time))
    write_paragraph("مدة الطهي", str(cooking_time))

    # === Save DOCX ===
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    docx_path = os.path.join(DOCX_DIR, f"{safe_name}.docx")
    pdf_path = os.path.join(PDF_DIR, f"{safe_name}.pdf")
    doc.save(docx_path)

    # === Convert to PDF ===
    try:
        convert(docx_path, pdf_path)
        print(f"✅ Converted to PDF: {pdf_path}")
    except Exception as e:
        print(f"❌ Error converting {safe_name} to PDF: {e}")

print("✅ All restaurants converted to both Word and PDF formats.")
