import json
import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import arabic_reshaper
from bidi.algorithm import get_display

# === Settings ===
INPUT_FILE = "Resturants.json"
OUTPUT_DIR = "Resturants"
FONT_PATH = "Cairo/static/Cairo-Regular.ttf"  # Adjust if the path is different

# === Create output directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Arabic Text Fix ===
def reshape_arabic(text):
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except:
        return str(text)

# === PDF Writer Class ===
class SimplePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.add_font("Arabic", fname=FONT_PATH, uni=True)
        self.set_font("Arabic", size=14)
        self.set_right_margin(10)
        self.set_left_margin(10)

    def write_line(self, label, value):
        if isinstance(value, list):
            value = "، ".join(value)
        elif isinstance(value, dict):
            value = "\n".join([f"{k}: {v}" for k, v in value.items()])
        text = f"{label}: {value}"
        self.cell(0, 10, reshape_arabic(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')

# === Load JSON ===
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === Generate PDFs ===
for rest in data:
    name = rest.get("name", "مطعم بدون اسم")
    address = rest.get("adress", "")
    phone = rest.get("phone", [])
    comments = rest.get("comments", {})
    emenu = rest.get("eMenu", {})
    open_time = rest.get("openTime", "غير معروف")
    close_time = rest.get("closeTime", "غير معروف")
    cooking_time = rest.get("cookingTimeRange", "غير محدد")

    pdf = SimplePDF()
    pdf.write_line("اسم المطعم", name)
    pdf.write_line("العنوان", address)
    pdf.write_line("أرقام الهاتف", phone)

    # Comments
    pdf.write_line("التعليقات", "")
    for email, comment in comments.items():
        pdf.write_line(f"من {email}", comment)

    # eMenu
    pdf.write_line("القائمة الإلكترونية", "")
    for section_name, section in emenu.items():
        pdf.write_line("القسم", section_name)
        if 'products' in section:
            for product in section['products']:
                pdf.write_line(" - الاسم", product.get("name", ""))
                pdf.write_line("   الوصف", product.get("desc", ""))
                pdf.write_line("   السعر", product.get("price", ""))
                if 'sizes' in product:
                    for size, price in product['sizes'].items():
                        pdf.write_line(f"   الحجم: {size}", price)
                if 'extras' in product:
                    for extra, extra_price in product['extras'].items():
                        pdf.write_line(f"   إضافة: {extra}", extra_price)

    pdf.write_line("وقت الفتح", str(open_time))
    pdf.write_line("وقت الإغلاق", str(close_time))
    pdf.write_line("مدة الطهي", str(cooking_time))

    # Save PDF
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    filename = f"{OUTPUT_DIR}/{safe_name}.pdf"
    pdf.output(filename)

print("✅ Arabic PDFs created successfully with proper text shaping and direction!")
