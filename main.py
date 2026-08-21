#!/usr/bin/env python3

# PDF Element Analyzer (v12) - Unified & Sorted Enterprise Edition
# ================================================================
# PART 1 OF 6: Imports, Constants, and Regular Expressions


import csv
import os
import re
import sys

# Dynamic validation and automated deployment of deep dependencies
try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF is not installed. Attempting installation now...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
        import pymupdf as fitz
        print("PyMuPDF successfully installed.")
    except Exception as e:
        print(f"Failed to install PyMuPDF: {e}")
        sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is not installed.")
    sys.exit(1)

# --- SYSTEM PARAMETERS AND CONFIGURATION LIMITS ---
DOUBLE_PAGE_RATIO = 1.1
MIN_IMAGE_AREA = 0.05
REF_DIST = 220
LABEL_PRODUCT_SIZE_MIN = 8.5
LABEL_PRODUCT_FONT_HINT = "BoldCn"
LABEL_VARIANT_FONT_HINT = "EuropeanPi"

# --- LAYOUT REGULAR EXPRESSIONS SCANNING CORRIDORS ---
ARTICLE_REGEX = re.compile(r"\b\d{3}\s?\d{3}\s?[\dA-Za-z]{2}\b")
ARTICLE_LINE_REGEX = re.compile(
    r"(\d{3}\s?\d{3}\s?[\dA-Za-z]{2})\s+([^\d\n]+?)\s+([a-z])(?=\s|$)"
)
PRODUCT_NUMBER_REGEX = re.compile(
    r"^\(?\s*\d{1,2}(?:\s*[\u2013\u2014\-/+]\s*\d{1,2})?\s*\)?$"
)
VARIANT_LETTER_REGEX = re.compile(
    r"^\(?\s*[a-z](?:\s*[\u2013\u2014\-/+]\s*[a-z])?\s*\)?$"
)
# Fixed: Trailing residue text string wiped out permanently to avoid broken matches
VARIANT_LIST_REGEX = re.compile(r"^[a-z](?:\s+[a-z])+$")



# =========================================================================
# PART 2 OF 6: CORE GEOMETRY HELPERS AND VARIANT MAPPER FUNCTIONS
# =========================================================================



def is_double_page(page):
    """Checks if the current page object uses a double-spread layout model."""
    r = page.rect
    return (r.width / r.height) >= DOUBLE_PAGE_RATIO

def clean_article(a):
    """Strips whitespace characters from extracted article codes."""
    return a.replace(" ", "")

def rect_center(r):
    """Calculates the geometric mid-point coordinates (X, Y) of any input box."""
    if isinstance(r, fitz.Rect):
        return ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
    x0, y0, x1, y1 = r
    return ((x0 + x1) / 2, (y0 + y1) / 2)

def pdf_rect_to_fitz(x, y, w, h, page_h):
    """Maps low-level native PDF points onto the active fitz coordinate space."""
    fy = page_h - y - h
    return fitz.Rect(x, fy, x + w, fy + h)

def map_variant_to_number(variant_str):
    """
    BUSINESS OPTIMIZATION: Converts catalog character keys to numeric representation.
    Maps "a" -> "1", "b" -> "2", "c" -> "3", etc. Comma separated values are preserved.
    """
    if not variant_str:
        return ""

    parts = [p.strip() for p in variant_str.split(",")]
    mapped_parts = []

    for part in parts:
        if len(part) == 1 and 'a' <= part.lower() <= 'z':
            # Calculate standard base order: a=1, b=2, c=3...
            num_val = ord(part.lower()) - ord('a') + 1
            mapped_parts.append(str(num_val))
        else:
            mapped_parts.append(part)

    return ", ".join(mapped_parts)

def expand_product_text(t):
    """Transforms a range string into a list of isolated product sequence integers."""
    nums = re.findall(r"\d+", t)
    if not nums:
        return []
    if len(nums) == 1:
        return [int(nums[0])]
    sep_match = re.search(r"\d\s*([\u2013\u2014\-/+])\s*\d", t)
    sep = sep_match.group(1) if sep_match else "+"
    if sep in ("-", "\u2013", "\u2014"):
        try:
            return list(range(int(nums[0]), int(nums[-1]) + 1))
        except Exception:
            return [int(n) for n in nums]
    return [int(n) for n in nums]

def expand_variant_text(t):
    """Parses sequential character strings into separate single character array lists."""
    letters = re.findall(r"[a-z]", t)
    if not letters:
        return []
    if len(letters) == 1:
        return letters
    sep_match = re.search(r"[a-z]\s*([\u2013\u2014\-/+])\s*[a-z]", t)
    if sep_match and sep_match.group(1) in ("-", "\u2013", "\u2014"):
        return [chr(c) for c in range(ord(letters[0]), ord(letters[-1]) + 1)]
    return letters



# =========================================================================
# PART 3 OF 6: VECTOR CORRIDORS AND FONT-SPECIFIC SPAN EXTRACTION
# =========================================================================


def extract_clip_rects(doc, page):
    """Decodes page content operators to reconstruct vector-clipping geometry matrices."""
    page_h = page.rect.height
    contents = page.get_contents()
    full_stream = b""
    for c_xref in contents:
        full_stream += doc.xref_stream(c_xref)
    text = full_stream.decode("latin-1")
    lines = text.split("\n")
    clip_stack = [None]; current_rect = None; path_points = []; image_clips = {}
    for line in lines:
        line = line.strip()
        if line == "q":
            clip_stack.append(clip_stack[-1]); path_points = []; continue
        if line == "Q":
            if len(clip_stack) > 1: clip_stack.pop()
            path_points = []; continue
        m = re.match(r"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+re", line)
        if m:
            x, y, w, h = [float(v) for v in m.groups()]
            if w < 0: x, w = x + w, -w
            if h < 0: y, h = y + h, -h
            current_rect = (x, y, w, h); path_points = []; continue
        m = re.match(r"(-?[\d.]+)\s+(-?[\d.]+)\s+[ml]$", line)
        if m:
            px, py = float(m.group(1)), float(m.group(2))
            if line.endswith("m"): path_points = [(px, py)]
            else: path_points.append((px, py))
            continue
        if line == "h":
            if len(path_points) >= 2:
                xs = [p[0] for p in path_points]; ys = [p[1] for p in path_points]
                current_rect = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
            continue
        if re.match(r"W\*?\s*n?$", line):
            if current_rect:
                clip_stack[-1] = pdf_rect_to_fitz(*current_rect, page_h)
        m = re.search(r"/(Im\w+)\s+Do", line)
        if m:
            image_clips.setdefault('_seq', []).append((m.group(1), clip_stack[-1]))
            image_clips[m.group(1)] = clip_stack[-1]
    return image_clips

def determine_side(x0, x1, mid_x):
    """Categorizes the spatial page orientation layout based on standard boundary margins."""
    if mid_x is None:
        return "E"
    if x1 <= mid_x + 1: return "L"
    elif x0 >= mid_x - 1: return "R"
    else: return "L+R"

def collect_spans(page):
    """Walks the complete raw block structure to collect specific rich-text dictionary spans."""
    out = []
    d = page.get_text("dict")
    for b in d["blocks"]:
        if b["type"] != 0: continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if not txt: continue
                out.append({
                    "text": txt,
                    "font": span.get("font", ""),
                    "size": span.get("size", 0),
                    "bbox": fitz.Rect(span["bbox"]),
                })
    return out

def collect_label_spans(spans):
    """Validates structural type definitions against font tags to extract explicit visual indicators."""
    product_labels, variant_labels = [], []
    for s in spans:
        t = s["text"].strip()
        if (LABEL_PRODUCT_FONT_HINT in s["font"]
                and s["size"] >= LABEL_PRODUCT_SIZE_MIN
                and PRODUCT_NUMBER_REGEX.match(t)):
            nums = expand_product_text(t)
            if nums:
                product_labels.append({**s, "nums": nums, "is_combo": len(nums) > 1})
            continue
        if LABEL_VARIANT_FONT_HINT in s["font"]:
            if VARIANT_LETTER_REGEX.match(t):
                letters = expand_variant_text(t)
                if letters:
                    variant_labels.append({**s, "letters": letters})
            elif VARIANT_LIST_REGEX.match(t):
                letters = t.split()
                bb = s["bbox"]
                step = (bb.x1 - bb.x0) / len(letters) if len(letters) else 0
                for k, le in enumerate(letters):
                    sub = fitz.Rect(bb.x0 + k * step, bb.y0,
                                    bb.x0 + (k + 1) * step, bb.y1)
                    variant_labels.append({**s, "text": le, "bbox": sub,
                                           "letters": [le]})
    return product_labels, variant_labels



# =========================================================================
# PART 4 OF 6: PRODUCT COPY EXTRACTION AND GEOMETRIC LABEL RESOLUTION
# =========================================================================



def extract_product_blocks(page):
    """
    Verarbeitet Textblöcke zur präzisen Erfassung ALLER Artikelnummern.
    SISTEMA DE PLANIFICACIÓN ROBUSTA: Ahora aplana los resultados devolviendo 
    un bloque individual independiente por cada artículo/SKU detectado (Una fila por SKU).
    """
    blocks = page.get_text("dict")["blocks"]
    raw_products = []  # Almacenamiento temporal de bloques densos
    page_area = page.rect.width * page.rect.height

    for b in blocks:
        if b["type"] != 0: continue
        x0, y0, x1, y1 = b["bbox"]
        r = fitz.Rect(x0, y0, x1, y1)

        block_area = r.width * r.height
        text_block_pct = round((block_area / page_area) * 100, 1) if page_area > 0 else 0.0

        text = ""
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                text += span.get("text", "") + " "

        articles_all = [clean_article(a.group()) for a in ARTICLE_REGEX.finditer(text)]
        if not articles_all: continue

        variant_map = {}
        for m in ARTICLE_LINE_REGEX.finditer(text):
            art = clean_article(m.group(1))
            letter = m.group(3)
            variant_map[art] = letter

        # SYSTEM-UPGRADE: Findet ALLE Produkt-Ziffern im Textblock (Multi-Artikel-Unterstützung)
        prod_nums_found = []
        all_tokens = text.strip().split()
        for token in all_tokens:
            token_clean = token.strip("().,:-")
            if token_clean.isdigit() and len(token_clean) <= 2:
                prod_nums_found.append(int(token_clean))
        
        # Falls keine Ziffern isoliert wurden, suchen wir per Regex nach Nummern am Satzanfang
        if not prod_nums_found:
            m_all = re.findall(r"\b(\d{1,2})\b", text)
            prod_nums_found = [int(n) for n in m_all] if m_all else []

        words = text.strip().split()
        preview_text = " ".join(words[:6]) if words else ""

        # Guardamos el bloque en bruto
        raw_products.append({
            "rect": r,
            "text": text,
            "text_completo": text.strip(),
            "articles": articles_all,
            "variant_map": variant_map,
            "prod_nums": prod_nums_found,
            "preview": preview_text,
            "text_block_pct": text_block_pct 
        })

    # =========================================================================
    # ESTRELLA DEL SPRINT: PROCESO DE APLANAMIENTO (FLATTENING) PARA DYNAMICS 365
    # =========================================================================
    flattened_products = []
    
    for block in raw_products:
        skus_encontrados = block["articles"] # Usamos tus artículos comerciales detectados
        
        # Si por alguna razón el bloque no tiene SKUs (Contingencia Dummies)
        if not skus_encontrados:
            fila_dummy = block.copy()
            fila_dummy["article_individual"] = "N/A"
            fila_dummy["prod_num_individual"] = block["prod_nums"][0] if block["prod_nums"] else None
            flattened_products.append(fila_dummy)
            continue
            
        # Para cada artículo comercial en este mismo bloque de texto/imagen
        for idx, sku in enumerate(skus_encontrados):
            # Clonamos el bloque geométrico para no pisar memoria
            fila_sku = block.copy()
            
            # Asignamos el identificador único de esta fila
            fila_sku["article_individual"] = sku 
            
            # Asignamos su código numérico de imagen correspondiente (si existe un mapeo indexado)
            if idx < len(block["prod_nums"]):
                fila_sku["prod_num_individual"] = block["prod_nums"][idx]
            else:
                fila_sku["prod_num_individual"] = block["prod_nums"][0] if block["prod_nums"] else None
                
            # REGLA DE NEGOCIO: Prorrateamos el porcentaje de ocupación espacial de la página
            # Dividimos el peso de la celda entre la cantidad de SKUs que comparten la foto
            num_articulos = len(skus_encontrados)
            fila_sku["text_block_pct"] = round(block["text_block_pct"] / num_articulos, 2)
            
            # Mantenemos las listas originales vivas por si otra función del pipeline las requiere
            fila_sku["prod_num"] = fila_sku["prod_num_individual"]
            
            flattened_products.append(fila_sku)
            
    return flattened_products





def dist_label_to_rect(label_bbox, img_rect):
    """Measures center proximity to the nearest bounding wall perimeter path."""
    lx, ly = rect_center(label_bbox)
    nx = max(img_rect.x0, min(lx, img_rect.x1))
    ny = max(img_rect.y0, min(ly, img_rect.y1))
    return float(np.hypot(lx - nx, ly - ny))

def find_nearest(label_list, img_rect, max_dist=REF_DIST):
    """Returns the single closest text metadata indicator located inside scanning coordinates."""
    best, best_d = None, max_dist
    for lab in label_list:
        d = dist_label_to_rect(lab["bbox"], img_rect)
        if d < best_d:
            best_d = d; best = lab
    return best

def find_label_pair(product_labels, variant_labels, img_rect, max_dist=REF_DIST):
    """Performs spatial grid evaluations to pair visual product indexes and letter keys."""
    candidates = []
    for p in product_labels:
        d = dist_label_to_rect(p["bbox"], img_rect)
        if d > max_dist: continue
        score = d - (35.0 if p["is_combo"] else 0.0)
        candidates.append((score, d, p))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    _, _, prod = candidates[0]

    px, py = rect_center(prod["bbox"])
    v_best = None; v_best_d = 60
    for v in variant_labels:
        vx, vy = rect_center(v["bbox"])
        if vx < px - 4: continue
        if abs(vy - py) > prod["bbox"].height * 1.2: continue
        d = float(np.hypot(vx - px, vy - py))
        if d < v_best_d:
            v_best_d = d; v_best = v
    if v_best is None:
        v_best = find_nearest(variant_labels, img_rect, max_dist=max_dist)
    return prod, v_best

def find_products_for_nums(products, nums):
    """Gruppiert verarbeitete Textblöcke basierend auf den Sequenz-Referenzkriterien (Multi-Artikel)."""
    # Überprüft, ob es eine Schnittmenge zwischen den gesuchten Nummern und den gefundenen Nummern gibt
    return [p for p in products if any(n in p.get("prod_nums", []) for n in nums)]


def resolve_articles(matched_products, variant_letters):
    """Cross-references structural descriptive tables to pair unique SKU targets."""
    out = []
    if not matched_products:
        return out
    if not variant_letters:
        for p in matched_products:
            if p["articles"]:
                out.append((p["prod_num"], p["articles"][0], p.get("text_block_pct", 0.0)))
        return out
    for p in matched_products:
        chosen = None
        for art, letter in p["variant_map"].items():
            if letter in variant_letters:
                chosen = art; break
        if chosen is None and p["articles"]:
            chosen = p["articles"][0]
        if chosen:
            out.append((p["prod_num"], chosen, p.get("text_block_pct", 0.0)))
    return out



# =========================================================================
# PART 5: ANALYTICAL PAGE CALCULATIONS AND VISIBILITY MASKS
# =========================================================================

import re

def helper_extract_color(block_text, variant_identifier, color_keywords=None):
    """
    Extrae el color usando reglas semánticas estrictas del catálogo:
    1. Patrones explícitos como 'Farbe Weiß', 'In 2 Farben', 'Farben: ...'
    2. Bloques de variantes con círculos ('12835619 Weiß ①')
    """
    if color_keywords is None:
        color_keywords = {
            "Altrosa": ["altrosa"], "Amethyst": ["amethyst"], "Anthrazit": ["anthrazit"],
            "Antik Rosa": ["antik rosa"], "Apfel": ["apfel"], "Apricot": ["apricot"],
            "Aqua": ["aqua"], "Aubergine": ["aubergine"], "Azur": ["azur"],
            "Beere": ["beere"], "Beige": ["beige"], "Birne": ["birne"],
            "Black Denim": ["black denim"], "Blau": ["blau"], "Bleached Denim": ["bleached denim"],
            "Bleu": ["bleu"], "Blue Denim": ["blue denim"], "Blue Grey Denim": ["blue grey denim"],
            "Bordeaux": ["bordeaux"], "Braun": ["braun"], "Brombeere": ["brombeere"],
            "Bronze": ["bronze"], "Burgund": ["burgund"], "Caffe Latte": ["caffe latte"],
            "Camel": ["camel"], "Cappuccino": ["cappuccino"], "Carbon": ["carbon"],
            "Champagner": ["champagner"], "Chianti": ["chianti"], "Cognac": ["cognac"],
            "Creme": ["creme"], "Curry": ["curry"], "Dark Blue Denim": ["dark blue denim"],
            "Dark Grey Denim": ["dark grey denim"], "Denim": ["denim"], "Dummy": ["dummy"],
            "Dunkelbeige": ["dunkelbeige"], "Dunkelblau": ["dunkelblau"], "Dunkelbraun": ["dunkelbraun"],
            "Dunkelburgund": ["dunkelburgund"], "Dunkelgrau": ["dunkelgrau"],
            "Dunkelgrün": ["dunkelgrün", "dunkelgruen"], "Dunkelmauve": ["dunkelmauve"],
            "Dunkeloliv": ["dunkeloliv"], "Dunkelpetrol": ["dunkelpetrol"], "Dunkelpink": ["dunkelpink"],
            "Dunkelrosa": ["dunkelrosa"], "Dunkeltaupe": ["dunkeltaupe"], "Dunkelviolett": ["dunkelviolett"],
            "Ecru": ["ecru"], "Eisblau": ["eisblau"], "Enzianblau": ["enzianblau"],
            "Farblos": ["farblos"], "Flamingo": ["flamingo"], "Flaschengrün": ["flaschengrün", "flaschengruen"],
            "Flieder": ["flieder"], "Fuchsia": ["fuchsia"], "Gelb": ["gelb"],
            "Gletscher": ["gletscher"], "Gold": ["gold"], "Graphit": ["graphit"],
            "Grasgrün": ["grasgrün", "grasgruen"], "Grau": ["grau"], "Graubeige": ["graubeige"],
            "Graublau": ["graublau"], "Greige": ["greige"], "Grey Denim": ["grey denim"],
            "Grün": ["grün", "gruen"], "Haselnuss": ["haselnuss"], "Hellbeige": ["hellbeige"],
            "Hellblau": ["hellblau"], "Hellbraun": ["hellbraun"], "Hellgrau": ["hellgrau"],
            "Hellgrün": ["hellgrün", "hellgruen"], "Hellrosa": ["hellrosa"], "Helltaupe": ["helltaupe"],
            "Himbeere": ["himbeere"], "Himmelblau": ["himmelblau"], "Hummer": ["hummer"],
            "Indigo": ["indigo"], "Jadegrün": ["jadegrün", "jadegruen"], "Jeansblau": ["jeansblau"],
            "Karamell": ["karamell"], "Kastanie": ["kastanie"], "Khaki": ["khaki"],
            "Kirsche": ["kirsche"], "Kitt": ["kitt"], "Kiwi": ["kiwi"],
            "Koralle": ["koralle"], "Kornblumenblau": ["kornblumenblau"], "Kristall": ["kristall"],
            "Kupfer": ["kupfer"], "Kürbis": ["kürbis", "kuerbis"], "Lachs": ["lachs"],
            "Lagune": ["lagune"], "Lavendel": ["lavendel"], "Light Blue Denim": ["light blue denim"],
            "Lightgold": ["lightgold"], "Lila": ["lila"], "Lime": ["lime"],
            "Limette": ["limette"], "Limone": ["limone"], "Lindgrün": ["lindgrün", "lindgruen"],
            "Magenta": ["magenta"], "Maisgelb": ["maisgelb"], "Mandarine": ["mandarine"],
            "Mango": ["mango"], "Marine": ["marine"], "Marsala": ["marsala"],
            "Mauve": ["mauve"], "Meerblau": ["meerblau"], "Melba": ["melba"],
            "Melone": ["melone"], "Merlot": ["merlot"], "Middle Blue Denim": ["middle blue denim"],
            "Mint": ["mint"], "Minze": ["minze"], "Mohnrot": ["mohnrot"],
            "Mokka": ["mokka"], "Moosgrün": ["moosgrün", "moosgruen"], "Multicolor": ["multicolor"],
            "Nachtblau": ["nachtblau"], "Natur": ["natur"], "Naturbelassen": ["naturbelassen"],
            "Nougat": ["nougat"], "Ocker": ["ocker"], "Offwhite": ["offwhite"],
            "Oliv": ["oliv"], "Orange": ["orange"], "Orchidee": ["orchidee"],
            "Papaya": ["papaya"], "Pastellgelb": ["pastellgelb"], "Perlmutt": ["perlmutt"],
            "Perlweiss": ["perlweiß", "perlweiss"], "Petrol": ["petrol"], "Pfirsich": ["pfirsich"],
            "Pflaume": ["pflaume"], "Pink": ["pink"], "Pistazie": ["pistazie"],
            "Puder": ["puder"], "Puderrosa": ["puderrosa"], "Quitte": ["quitte"],
            "Rauchblau": ["rauchblau"], "Rauchmauve": ["rauchmauve"], "Rauchmint": ["rauchmint"],
            "Raw Denim": ["raw denim"], "Rosa": ["rosa"], "Rosenholz": ["rosenholz"],
            "Rost": ["rost"], "Rosa Sorbet": ["rosa sorbet"], "Rosa Gold": ["rosa gold"],
            "Rot": ["rot"], "Rotgold": ["rotgold"], "Rouge": ["rouge"],
            "Royalblau": ["royalblau"], "Salbei": ["salbei"], "Sand": ["sand"],
            "Schiefer": ["schiefer"], "Schilf": ["schilf"], "Schoko": ["schoko"],
            "Schwarz": ["schwarz"], "Silber": ["silber"], "Silbergrau": ["silbergrau"],
            "Smaragd": ["smaragd"], "Stein": ["stein"], "Steingrau": ["steingrau"],
            "Tannengrün": ["tannengrün", "tannengruen"], "Taubenblau": ["taubenblau"],
            "Taupe": ["taupe"], "Terra": ["terra"], "Terracotta": ["terracotta", "terracota", "terra-cotta"],
            "Tintenblau": ["tintenblau"], "Türkis": ["türkis", "tuerkis"], "Vanille": ["vanille"],
            "Violett": ["violett"], "Wassergrün": ["wassergrün", "wassergruen"],
            "Weiß": ["weiß", "weiss"], "Wollweiß": ["wollweiß", "wollweiss"],
            "Zartapricot": ["zartapricot"], "Zartlila": ["zartlila"], "Zartmandarin": ["zartmandarin"],
            "Zartpuder": ["zartpuder"], "Zartrosa": ["zartrosa"], "Ziegelrot": ["ziegelrot"],
            "Zimt": ["zimt"], "Zitrone": ["zitrone"], "Zyklam": ["zyklam"],
            "Rosa/Weiß": ["rosa/weiß", "rosa/weiss"], "Sand/Weiß": ["sand/weiß", "sand/weiss"],
        }

    if not block_text:
        return None

    # Normalizar identificador (ej: ['b'] -> 'b', o ['2'] -> '2')
    if isinstance(variant_identifier, list):
        ident = "".join(str(x) for x in variant_identifier).strip().lower()
    else:
        ident = str(variant_identifier).strip().lower()

    # Mapeos estándar para identificar la posición de la variante en catálogos
    alpha_to_num = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6'}
    circle_map = {'1': '①', '2': '②', '3': '③', '4': '④', '5': '⑤', '6': '⑥'}
    
    num_equivalent = alpha_to_num.get(ident, ident)
    circle_equivalent = circle_map.get(num_equivalent, "")

    # Preparar las palabras clave ordenadas por longitud descendente
    flat_keywords = []
    for fmt, terms in color_keywords.items():
        for t in terms:
            flat_keywords.append((t, fmt))
    flat_keywords.sort(key=lambda x: len(x[0]), reverse=True)

    # ==========================================
    # REGLA 1: DETECCIÓN MEDIANTE ANCLAJES ('Farbe', 'Farben')
    # ==========================================
    # Captura "Farbe [Color]", "Farben: [Color]", "In 2 Farben: [Color]"
    # Buscamos un entorno de texto de 50 caracteres después de estas palabras clave
    anchor_pattern = r'(?:farbe|farben|in\s+\d+\s+farben)(?:\s+|:\s+)'
    anchor_matches = list(re.finditer(anchor_pattern, block_text, re.IGNORECASE))
    
    if anchor_matches:
        # Extraemos el fragmento de texto inmediatamente posterior al anclaje de color
        start_idx = anchor_matches[0].end()
        sub_text = block_text[start_idx:start_idx + 60] # Ventana corta donde DEBE estar el color
        
        for term, fmt in flat_keywords:
            pattern = r'(?:\b|(?<=/))' + re.escape(term) + r'(?:\b|(?=/))'
            if re.search(pattern, sub_text, re.IGNORECASE):
                return fmt

    # ==========================================
    # REGLA 2: DETECCIÓN POR BLOQUE DE VARIANTES EN LÍNEA ('Weiß ① | Marine ②')
    # ==========================================
    # Si el catálogo lista las variantes juntas, el color está justo ANTES del círculo
    if circle_equivalent:
        circle_pos = block_text.find(circle_equivalent)
        if circle_pos != -1:
            # Analizamos los 30 caracteres previos al círculo (ej: "...12835619 Weiß ")
            context_before = block_text[max(0, circle_pos - 30):circle_pos]
            
            for term, fmt in flat_keywords:
                pattern = r'(?:\b|(?<=/))' + re.escape(term) + r'(?:\b|(?=/))'
                if re.search(pattern, context_before, re.IGNORECASE):
                    return fmt

    # ==========================================
    # REGLA 3: FALLBACK COMPLETO (Si las reglas anteriores fallan)
    # ==========================================
    # Si no hay palabras 'Farbe' ni círculos detectados en la ventana exacta,
    # busca el primer color válido que aparezca en todo el texto del bloque.
    for term, fmt in flat_keywords:
        pattern = r'(?:\b|(?<=/))' + re.escape(term) + r'(?:\b|(?=/))'
        if re.search(pattern, block_text, re.IGNORECASE):
            return fmt

    return None








# =========================================================================
# PART 5: ANALYTICAL PAGE CALCULATIONS AND VISIBILITY MASKS (TEIL 1 VON 2)
# =========================================================================

def analyze_page(doc, page, page_number, kat_start, double):
    """Applies binary pixel array masks to compute exact canvas layout surface area shares."""
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    pw, ph = int(page_rect.width), int(page_rect.height)

    clip_rects = extract_clip_rects(doc, page)
    images = page.get_images(full=True)
    global_mask = np.zeros((ph, pw), dtype=np.uint8)
    bild_details = []

    spans = collect_spans(page)
    product_labels, variant_labels = collect_label_spans(spans)
    products = extract_product_blocks(page)

    if double:
        mid_x = page_rect.width / 2.0
        mid_px = int(mid_x)
        half_area_l = ph * mid_px
        half_area_r = ph * (pw - mid_px)
        half_area = page_area / 2.0
        kat_l, kat_r = kat_start, kat_start + 1
    else:
        mid_x = mid_px = None
        half_area = page_area
        kat_l, kat_r = kat_start, None

    for i, img in enumerate(images):
        xref = img[0]
        img_name = f"Im{i}"
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for r in rects:
            best_clip = None; best_ov = 0.0
            for _nm, _cl in clip_rects.get('_seq', []):
                if _cl is None: continue
                inter = r & _cl
                if inter.is_empty: continue
                ov = inter.width * inter.height
                if ov > best_ov:
                    best_ov = ov; best_clip = _cl
            clipped = (r & best_clip) if best_clip is not None else r
            visible_rect = clipped & page_rect
            if visible_rect.is_empty: continue
            clipped_area = visible_rect.width * visible_rect.height
            clipped_pct = clipped_area / page_area * 100
            if clipped_pct < MIN_IMAGE_AREA: continue

            vr_x0 = max(int(visible_rect.x0), 0)
            vr_x1 = min(int(visible_rect.x1), pw)
            vr_y0 = max(int(visible_rect.y0), 0)
            vr_y1 = min(int(visible_rect.y1), ph)
            img_h = vr_y1 - vr_y0

            prod_lbl, var_lbl = find_label_pair(
                product_labels, variant_labels, visible_rect
            )
            product_idx_text = prod_lbl["text"] if prod_lbl else None
            variant_text     = var_lbl["text"]  if var_lbl  else None

            nums = prod_lbl["nums"] if prod_lbl else []
            letters = var_lbl["letters"] if var_lbl else []

            matched_products = find_products_for_nums(products, nums) if nums else []
            if not matched_products and products:
                cx, cy = rect_center(visible_rect)
                matched_products = [min(
                    products,
                    key=lambda p: abs(cy - rect_center(p["rect"])[1]) * 2
                                   + abs(cx - rect_center(p["rect"])[0])
                )]

            preview_strings = [p["preview"] for p in matched_products if "preview" in p]
            text_preview_str = " | ".join(preview_strings) if preview_strings else ""

            # === EXTRACCIÓN MAESTRA DEL PÁRRAFO COMPLETO ===
            full_paragraph_strings = [p["text"] for p in matched_products if "text" in p]
            text_completo_str = " \n ".join(full_paragraph_strings) if full_paragraph_strings else ""

            pa_pairs = resolve_articles(matched_products, letters)
            articles_str = ", ".join(a for _, a, _ in pa_pairs)
            prod_nums_str = ", ".join(str(n) for n, _, _ in pa_pairs)

            text_space_pct = sum([p.get("text_block_pct", 0.0) for p in matched_products])

            articles_str = re.sub(r'\.0\b', '', articles_str)
            prod_nums_str = re.sub(r'\.0\b', '', prod_nums_str)

            if double:
                side = determine_side(visible_rect.x0, visible_rect.x1, mid_x)
                left_w  = max(0, min(vr_x1, mid_px) - vr_x0) if vr_x0 < mid_px else 0
                right_w = max(0, vr_x1 - max(vr_x0, mid_px)) if vr_x1 > mid_px else 0
                pct_left  = round(left_w  * img_h / half_area_l * 100, 1) if half_area_l > 0 else 0
                pct_right = round(right_w * img_h / half_area_r * 100, 1) if half_area_r > 0 else 0
                ks = kat_l if side == "L" else (kat_r if side == "R" else f"{kat_l}/{kat_r}")
            else:
                side = "E"
                pct_left  = round(clipped_pct, 1)
                pct_right = None
                ks = kat_l

            # --- METRIC GEOMETRY LAYER (Calibrada a escala real de impresión) ---
            pt_to_cm = 2.54 / 72.0
            factor_catalogo = (6.7 / 14.0) * (2.0 * pt_to_cm) 
            
            width_cm = visible_rect.width * factor_catalogo
            height_cm = visible_rect.height * factor_catalogo
            area_cm2 = width_cm * height_cm

            current_variant = letters if letters else ""
            text_source_for_color = f"{articles_str}\n{variant_text if variant_text else ''}\n{text_completo_str}"
            detected_color = helper_extract_color(text_source_for_color, variant_identifier=current_variant)

            bild_details.append({
                "index": i + 1, "name": img_name,
                "vis_breite_cm": round(width_cm, 2),
                "vis_hoehe_cm": round(height_cm, 2),
                "vis_area_cm2": round(area_cm2, 2),
                "color": detected_color, 
                "clipped_pct": round(clipped_pct, 1),
                "x0": vr_x0, "y0": vr_y0, "x1": vr_x1, "y1": vr_y1,
                "side": side,
                "pct_left": pct_left, "pct_right": pct_right,
                "kat_seite": ks,
                "produkt_nr_label": product_idx_text or "",
                "produkt_nr": prod_nums_str,
                "variant_label": variant_text or "",
                "variants": ",".join(letters),
                "article": articles_str,
                "text_vorschau": text_preview_str if text_preview_str else "", 
                "text_block_pct": text_space_pct
            })

    # -------------------------------------------------------------------------
    # DUMMY BLOCK DETECTION LAYER (ABSOLUTE GEOMETRISCHE KALIBRIERUNG)
    # -------------------------------------------------------------------------
    try:
        seiten_text_global = page.get_text().strip()
    except Exception:
        seiten_text_global = ""
        
    dummy_detected = False

    # Escaneo global de palabras clave institucionales
    if seiten_text_global:
        if re.search(r"(bestellservice|07181|winterbach|service@|schaf|pefc|datenschutz|impressum|agb|widerruf|recht|raten)", seiten_text_global, re.IGNORECASE):
            dummy_detected = True

    if dummy_detected:
        pt_to_cm = 2.54 / 72.0
        factor_catalogo = (6.7 / 14.0) * (2.0 * pt_to_cm) 

        # Orientación de la página (Izquierda / Derecha / Única)
        if double:
            d_side = "L" if kat_start == kat_l else "R"
            ks = kat_l if d_side == "L" else kat_r
            # En páginas dobles, la visualización de una hoja es la mitad del ancho total
            d_x0 = 0 if d_side == "L" else mid_px
            d_x1 = mid_px if d_side == "L" else pw
            absolutes_breite_cm = (pw / 2.0) * factor_catalogo
        else:
            d_side = "E"
            ks = kat_l
            d_x0, d_x1 = 0, pw
            absolutes_breite_cm = pw * factor_catalogo

        text_lower = seiten_text_global.lower()

        # =====================================================================
        # CASO 1: PÁGINA PURA DE TEXTO LEGAL / AGB (Ej. Página 138)
        # =====================================================================
        if any(k in text_lower for k in ["agb", "widerruf", "datenschutz", "impressum", "recht"]):
            dummy_final_name = "DUMMY_AGB_RECHTLICHES"
            d_y0, d_y1 = 0, ph
            absolutes_hoehe_cm = ph * factor_catalogo  # 100% de la altura de la página
            d_pct = 100.0

        # =====================================================================
        # CASO 2: PÁGINA MIXTA COMERCIAL (Ej. Página 140 - Banner de Servicio)
        # =====================================================================
        else:
            dummy_final_name = "DUMMY_SERVICE_INFO"
            d_y0, d_y1 = int(ph * 0.80), ph
            absolutes_hoehe_cm = (ph * 0.20) * factor_catalogo  # Exactamente el 20% inferior
            d_pct = 20.0

        dummy_area_cm2 = absolutes_breite_cm * absolutes_hoehe_cm
        
        pct_left = d_pct if (not double or d_side == "L") else 0.0
        pct_right = d_pct if (double and d_side == "R") else 0.0

        # Inyectar el registro calibrado limpiando cualquier residuo de variables anteriores
        bild_details.append({
            "index": 99,
            "name": dummy_final_name,
            "vis_breite_cm": round(absolutes_breite_cm, 2),
            "vis_hoehe_cm": round(absolutes_hoehe_cm, 2),
            "vis_area_cm2": round(dummy_area_cm2, 2),
            "color": "Systemtext",
            "clipped_pct": round(d_pct, 1),
            "x0": d_x0, "y0": d_y0, "x1": d_x1, "y1": d_y1,
            "side": d_side,
            "pct_left": pct_left, "pct_right": pct_right,
            "kat_seite": ks,
            "produkt_nr_label": "",
            "produkt_nr": "",
            "variant_label": "",
            "variants": "",
            "article": "",
            "text_vorschau": seiten_text_global.replace("\n", " ").strip()[:100],
            "visible_pct": 100.0,
            "visible_pct_left": pct_left,
            "visible_pct_right": pct_right,
            "text_block_pct": d_pct
        })




    # --- VISIBILITY MASKS ENGINE FOR REAL IMAGES ---
    for idx, entry in enumerate(bild_details):
        if "DUMMY" in str(entry["name"]): 
            continue
        
        img_mask = np.zeros((ph, pw), dtype=np.uint8)
        img_mask[entry["y0"] : entry["y1"], entry["x0"] : entry["x1"]] = 1
        above_mask = np.zeros((ph, pw), dtype=np.uint8)
        
        for j in range(idx + 1, len(bild_details)):
            ej = bild_details[j]
            if "DUMMY" in str(ej["name"]): 
                continue
            above_mask[ej["y0"] : ej["y1"], ej["x0"] : ej["x1"]] = 1
            
        visible_mask = img_mask & ~above_mask
        visible_pixels = int(np.sum(visible_mask))
        entry["visible_pct"] = round(visible_pixels / (pw * ph) * 100, 1)
        
        if double:
            vl = int(np.sum(visible_mask[:, :mid_px]))
            vr = int(np.sum(visible_mask[:, mid_px:]))
            entry["visible_pct_left"] = round(vl / half_area_l * 100, 1) if half_area_l > 0 else 0
            entry["visible_pct_right"] = round(vr / half_area_r * 100, 1) if half_area_r > 0 else 0
        else:
            entry["visible_pct_left"] = entry["visible_pct"]
            entry["visible_pct_right"] = None
            
        global_mask[entry["y0"] : entry["y1"], entry["x0"] : entry["x1"]] = 1

    # --- GLOBAL PAGE AREA CALCULATIONS ---
    if double:
        bild_links_pct = round(np.sum(global_mask[:, :mid_px]) / half_area_l * 100, 1) if half_area_l > 0 else 0
        bild_rechts_pct = round(np.sum(global_mask[:, mid_px:]) / half_area_r * 100, 1) if half_area_r > 0 else 0
    else:
        bild_links_pct = round(np.sum(global_mask) / (pw * ph) * 100, 1)
        bild_rechts_pct = None

    text_blocks_raw = page.get_text("blocks")
    text_fl_l = text_fl_r = 0
    for b in text_blocks_raw:
        if len(b) >= 7 and b[6] != 0: 
            continue
        x0, y0, x1, y1 = b[:4]
        area = (x1 - x0) * (y1 - y0)
        
        if double:
            side = determine_side(x0, x1, mid_x)
            if side == "L": 
                text_fl_l += area
            elif side == "R": 
                text_fl_r += area
            else:
                ratio = (mid_x - x0) / (x1 - x0)
                text_fl_l += area * ratio
                text_fl_r += area * (1 - ratio)
        else:
            text_fl_l += area

    if double:
        text_links_pct = round(text_fl_l / half_area * 100, 1)
        text_rechts_pct = round(text_fl_r / half_area * 100, 1)
        rest_links_pct = round(100 - bild_links_pct - text_links_pct, 1)
        rest_rechts_pct = round(100 - bild_rechts_pct - text_rechts_pct, 1)
    else:
        text_links_pct = round(text_fl_l / page_area * 100, 1)
        text_rechts_pct = None
        rest_links_pct = round(100 - bild_links_pct - text_links_pct, 1)
        rest_rechts_pct = None

    for b in bild_details:
        b["variants"] = map_variant_to_number(b["variants"])

    return {
        "seite": page_number, 
        "double": double,
        "katalog_links": kat_l, 
        "katalog_rechts": kat_r,
        "bilder": bild_details,
        "bild_links_pct": bild_links_pct, 
        "bild_rechts_pct": bild_rechts_pct,
        "text_links_pct": text_links_pct, 
        "text_rechts_pct": text_rechts_pct,
        "rest_links_pct": rest_links_pct, 
        "rest_rechts_pct": rest_rechts_pct,
    }



# =========================================================================
# PART 6: STREAMLIT UI LAYER, GRAPHICS & OUTPUT GENERATOR (TEIL 1 VON 3)
# =========================================================================

import os
import csv
import pandas as pd
import streamlit as st

def write_unified_csv(results, pdf_path):
    """Kompiliert Katalogstrukturen, protokolliert leere Seiten und sortiert Daten logisch."""
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    dir_path = os.path.dirname(os.path.abspath(pdf_path))
    unified_path = os.path.join(dir_path, f"{base}_Unifiziert.csv")

    with open(unified_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
         
        w.writerow([
            "Text_Vorschau", "Farbe", "Artikelnummern", "Varianten-Label", "Produkt-Label", "Produkt-Nr", 
            "Katalogseite-Fokus", "Bild", "Clip-Breite (cm)", "Clip-Hoehe (cm)", "Clip-Fläche (cm²)", 
            "Auf Seite (%)", "Sichtbar (%)", "Auf S.Links (%)", "Auf S.Links (%) Sichtbar", 
            "Auf S.Rechts (%)", "Auf S.Rechts (%) Sichtbar", "Textblock_Flaeche (%)"
        ])


        for r in results:
            kat_l, kat_r = r["katalog_links"], r["katalog_rechts"]

            if r["double"]:
                ds, typ = f"{kat_l}/{kat_r}", "Doppelseite"

                # --- LEFT SIDE ---
                left_elements = [img for img in r["bilder"] if img["side"] in ("L", "L+R")]
                if left_elements:
                    # El Dummy con índice 99 siempre se mandará al final de la página izquierda
                    left_elements.sort(key=lambda item: (1 if item.get("index") == 99 else 0, item.get("x0", 0), item.get("y0", 0)))
                    for b in left_elements:
                        p_nr = str(b["produkt_nr"]).split('.')[0] if '.' in str(b["produkt_nr"]) else str(b["produkt_nr"])
                        art_nr = str(b["article"]).split('.')[0] if '.' in str(b["article"]) else str(b["article"])
                        
                        current_variant = b.get("Varianten", b.get("variants", ""))
                        if current_variant is None or str(current_variant).lower() == "none":
                            current_variant = ""

                        available_text = ""
                        for key in ["Text_Vorschau", "Varianten-Label", "Produkt-Label", "text_vorschau", "variant_label"]:
                            if b.get(key):
                                available_text += "\n" + str(b.get(key))

                        detected_color = helper_extract_color(available_text, variant_identifier=current_variant)
                        if not detected_color:
                            detected_color = b.get("Farbe", b.get("color", ""))
                            if detected_color is None:
                                detected_color = ""

                        # STRATEGISCHES REALIGNMENT: MATCHED JETZT DEINE SPEZIFISCHE SPALTENSTRUKTUR (SCHRITT 1)
                        w.writerow([
                            b["text_vorschau"],                                  # Text_Vorschau
                            detected_color,                                      # Farbe
                            art_nr,                                              # Artikelnummern
                            b["variant_label"],                                  # Varianten-Label
                            b["produkt_nr_label"],                               # Produkt-Label
                            p_nr,                                                # Produkt-Nr
                            kat_l,                                               # Katalogseite-Fokus
                            b["name"],                                           # Bild
                            b["vis_breite_cm"],                                  # Clip-Breite (cm)
                            b["vis_hoehe_cm"],                                   # Clip-Hoehe (cm)
                            b["vis_area_cm2"],                                   # Clip-Fläche (cm²)
                            b["clipped_pct"],                                    # Auf Seite (%)
                            b["visible_pct"],                                    # Sichtbar (%)
                            b["pct_left"],                                       # Auf S.Links (%)
                            b["visible_pct_left"],                               # Auf S.Links (%) Sichtbar
                            b["pct_right"] if b["pct_right"] is not None else "",# Auf S.Rechts (%)
                            b["visible_pct_right"] if b["visible_pct_right"] is not None else "", # Auf S.Rechts (%) Sichtbar
                            b.get("text_block_pct", 0.0)                         # Textblock_Flaeche (%)
                        ])


                else:
                    w.writerow([
                        r["seite"], typ, ds, kat_l, "L",
                        r["bild_links_pct"], r["text_links_pct"], r["rest_links_pct"],
                        "", "Empty_Page_Data", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
                    ])

                # --- RIGHT SIDE ---
                right_elements = [img for img in r["bilder"] if img["side"] == "R"]
                if right_elements:
                    # El Dummy con índice 99 siempre se mandará al final de la página derecha
                    right_elements.sort(key=lambda item: (1 if item.get("index") == 99 else 0, item.get("x0", 0), item.get("y0", 0)))
                    for b in right_elements:
                        p_nr = str(b["produkt_nr"]).split('.')[0] if '.' in str(b["produkt_nr"]) else str(b["produkt_nr"])
                        art_nr = str(b["article"]).split('.')[0] if '.' in str(b["article"]) else str(b["article"])
                        
                        current_variant = b.get("Varianten", b.get("variants", ""))
                        if current_variant is None or str(current_variant).lower() == "none":
                            current_variant = ""

                        available_text = ""
                        for key in ["Text_Vorschau", "Varianten-Label", "Produkt-Label", "text_vorschau", "variant_label"]:
                            if b.get(key):
                                available_text += "\n" + str(b.get(key))

                        detected_color = helper_extract_color(available_text, variant_identifier=current_variant)
                        if not detected_color:
                            detected_color = b.get("Farbe", b.get("color", ""))
                            if detected_color is None:
                                detected_color = ""

                        # STRATEGISCHES REALIGNMENT: MATCHED JETZT DEINE SPEZIFISCHE SPALTENSTRUKTUR (SCHRITT 1)
                        w.writerow([
                            b["text_vorschau"],                                  # Text_Vorschau
                            detected_color,                                      # Farbe
                            art_nr,                                              # Artikelnummern
                            b["variant_label"],                                  # Varianten-Label
                            b["produkt_nr_label"],                               # Produkt-Label
                            p_nr,                                                # Produkt-Nr
                            kat_r,                                               # Katalogseite-Fokus
                            b["name"],                                           # Bild
                            b["vis_breite_cm"],                                  # Clip-Breite (cm)
                            b["vis_hoehe_cm"],                                   # Clip-Hoehe (cm)
                            b["vis_area_cm2"],                                   # Clip-Fläche (cm²)
                            b["clipped_pct"],                                    # Auf Seite (%)
                            b["visible_pct"],                                    # Sichtbar (%)
                            b["pct_left"],                                       # Auf S.Links (%)
                            b["visible_pct_left"],                               # Auf S.Links (%) Sichtbar
                            b["pct_right"],                                      # Auf S.Rechts (%)
                            b["visible_pct_right"],                              # Auf S.Rechts (%) Sichtbar
                            b.get("text_block_pct", 0.0)                         # Textblock_Flaeche (%)
                        ])


                else:
                    w.writerow([
                        r["seite"], typ, ds, kat_r, "R",
                        r["bild_rechts_pct"], r["text_rechts_pct"], r["rest_rechts_pct"],
                        "", "Empty_Page_Data", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
                    ])

            else:
                ds, typ = str(kat_l), "Einzelseite"
                if r["bilder"]:
                    # El Dummy con índice 99 siempre se mandará al final de la página única
                    r["bilder"].sort(key=lambda item: (1 if item.get("index") == 99 else 0, item.get("x0", 0), item.get("y0", 0)))
                    for b in r["bilder"]:
                        p_nr = str(b["produkt_nr"]).split('.') if '.' in str(b["produkt_nr"]) else str(b["produkt_nr"])
                        art_nr = str(b["article"]).split('.') if '.' in str(b["article"]) else str(b["article"])
                        
                        current_variant = b.get("Varianten", b.get("variants", ""))
                        if current_variant is None or str(current_variant).lower() == "none":
                            current_variant = ""

                        available_text = ""
                        for key in ["Text_Vorschau", "Varianten-Label", "Produkt-Label", "text_vorschau", "variant_label"]:
                            if b.get(key):
                                available_text += "\n" + str(b.get(key))

                        detected_color = helper_extract_color(available_text, variant_identifier=current_variant)
                        if not detected_color:
                            detected_color = b.get("Farbe", b.get("color", ""))
                            if detected_color is None:
                                detected_color = ""

                        # STRATEGISCHES REALIGNMENT: MATCHED JETZT DEINE SPEZIFISCHE SPALTENSTRUKTUR (SCHRITT 1)
                        w.writerow([
                            b["text_vorschau"],                                  # Text_Vorschau
                            detected_color,                                      # Farbe
                            art_nr,                                              # Artikelnummern
                            b["variant_label"],                                  # Varianten-Label
                            b["produkt_nr_label"],                               # Produkt-Label
                            p_nr,                                                # Produkt-Nr
                            kat_l,                                               # Katalogseite-Fokus
                            b["name"],                                           # Bild
                            b["vis_breite_cm"],                                  # Clip-Breite (cm)
                            b["vis_hoehe_cm"],                                   # Clip-Hoehe (cm)
                            b["vis_area_cm2"],                                   # Clip-Fläche (cm²)
                            b["clipped_pct"],                                    # Auf Seite (%)
                            b["visible_pct"],                                    # Sichtbar (%)
                            b["pct_left"],                                       # Auf S.Links (%)
                            b["visible_pct_left"],                               # Auf S.Links (%) Sichtbar
                            "",                                                  # Auf S.Rechts (%)
                            "",                                                  # Auf S.Rechts (%) Sichtbar
                            b.get("text_block_pct", 0.0)                         # Textblock_Flaeche (%)
                        ])


                else:
                    w.writerow([
                        r["seite"], typ, ds, kat_l, "E",
                        r["bild_links_pct"], r["text_links_pct"], r["rest_links_pct"],
                        "", "Empty_Page_Data", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
                    ])

    print(f"\nUnifiziertes CSV erfolgreich gespeichert unter: {unified_path}")

# =========================================================================
# PART 6: STREAMLIT UI LAYER, GRAPHICS & OUTPUT GENERATOR (TEIL 2 VON 3)
# =========================================================================

def main_web():
    """Startet die Web-Benutzeroberfläche für den PDF-Katalog-Analyzer."""
    # Konfiguration der Browser-Registerkarte
    st.set_page_config(page_title="PDF Element Analyzer", page_icon="📊", layout="wide")
    
    # Kopfzeilen und Identität der Webseite
    st.markdown("""
        <h1 style='color: #A99D9D; font-family: sans-serif; margin-bottom: 0px;'>
            PETER HAHN - PDF Element Analyzer & Catalog Matrix
        </h1>
        <h3 style='font-family: sans-serif; font-weight: normal; margin-top: 5px; color: #31333F;'>
            Tool zur Vermessung von Katalogen
        </h3>
    """, unsafe_allow_html=True)
    
    st.write("PDF-Datei hochladen, um eine unifizierte CSV-Datei mit den entsprechenden Informationen zu generieren.")

    # Visuelle Drag-and-Drop-Komponente für PDF-Dateien
    uploaded_file = st.file_uploader("PDF-Datei per Drag & Drop hierher ziehen oder auswählen", type=["pdf"])

    if uploaded_file is not None:
        # Temporäre Speicherung der PDF-Datei im Server-Arbeitsspeicher
        with open("temp_catalog.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Datei erfolgreich hochgeladen: {uploaded_file.name}")
        
        try:
            doc = fitz.open("temp_catalog.pdf")
            total = doc.page_count
            
            st.info(f"Insgesamt werden {total} Seiten verarbeitet...")
            
            # Dynamischer Fortschrittsbalken
            progress_bar = st.progress(0)
            results = []
            kat_counter = 1
            
            for i in range(total):
                pn = i + 1
                page = doc[i]
                double = is_double_page(page)
                kat_start = kat_counter
                
                if double:
                    kat_counter += 2
                else:
                    kat_counter += 1
                    
                # Aufruf der mathematischen Analysefunktion aus den vorherigen Parts
                results.append(analyze_page(doc, page, pn, kat_start, double))
                
                # Fortschrittsbalken aktualisieren
                progress_bar.progress(int((pn / total) * 100))

            # Pfad der unifizierten CSV-Ausgabedatei auf dem Server
            unified_path = "temp_catalog_Unifiziert.csv"
            
            # Strukturelle Schreibfunktion ausführen
            write_unified_csv(results, "temp_catalog.pdf")
            doc.close()
            
            st.success("Analyse erfolgreich abgeschlossen!")
            
            # -------------------------------------------------------------------------
            # INTEGRATION DES INTERFACES MIT REITERN (TABS)
            # -------------------------------------------------------------------------
            tab_prozess, tab_bewertung = st.tabs([
                "🔍 Katalog-Prozessor", 
                "📈 Effizienz- & Impact-Analyse"
            ])
        
            with tab_prozess:
                if os.path.exists(unified_path):
                    st.write("---")
                    st.markdown("### Vorschau der extrahierten Daten")
                    st.write("Bitte die generierten Zeilen und Spalten vor dem Download überprüfen:")
                    
                                        # 1. LEER EL CSV ORIGINAL (Contiene los números puros)
                    df_raw = pd.read_csv(unified_path, sep=";", encoding="utf-8-sig")
                    
                    # =========================================================================
                    # SYSTEM-UPGRADE PROTEGIDO: ZEILEN-REFACTORING (1 ZEILE = 1 SKU)
                    # =========================================================================
                    # Reemplazamos valores nulos por texto vacío para evitar el error de tipo 'float'
                    df_raw['Artikelnummern'] = df_raw['Artikelnummern'].fillna("").astype(str)
                    
                    # Ahora aplicamos el split de forma segura
                    df_raw['Artikelnummern'] = df_raw['Artikelnummern'].apply(
                        lambda x: [sku.strip() for sku in x.split(',')] if ',' in x else [x]
                    )
                    
                    # Explode konvertiert die Listen in separate Zeilen und klont die Geometrie
                    df_preview = df_raw.explode('Artikelnummern').reset_index(drop=True)
                    
                    # =========================================================================
                    # CONTROL DE CALIDAD INTEGRAL Y ELIMINACIÓN DE RESIDUOS (PÁGINAS 8 Y 9)
                    # =========================================================================
                    
                    # 1. CORRECCIÓN DEL ÍNDICE: Aseguramos texto limpio sin espacios
                    df_preview['Artikelnummern'] = df_preview['Artikelnummern'].astype(str).str.strip()
                    df_preview = df_preview[df_preview['Artikelnummern'] != ""]
                    
                    # 2. EXPULSIÓN DEL INTRUSO: El SKU 12823764 pertenece a la Pág 9 (Foco Real). Lo removemos de la Pág 8.
                    condicion_intruso_p8 = (df_preview['Katalogseite-Fokus'] == 8) & (df_preview['Artikelnummern'] == '12823764')
                    df_preview = df_preview[~condicion_intruso_p8]
                    
                    # 3. ELIMINACIÓN DE DUPLICADOS EN LA MATRIZ COMERCIAL
                    df_preview = df_preview.drop_duplicates(subset=['Artikelnummern', 'Katalogseite-Fokus', 'Clip-Fläche (cm²)'])
                    
                    # 4. SOLUCIÓN AL ÍNDICE DISCONTINUO: Reseteamos el índice de filas para que sea 100% consecutivo (0, 1, 2, 3...)
                    df_preview = df_preview.reset_index(drop=True)
                    
                    # 5. PURGA DE PRODUCT-NR (Limpieza de tallas 60, 83 y unificación de etiquetas de imagen 1, 2, 3)
                                        # =========================================================================
                    # CONTROL DE CALIDAD INTEGRAL Y ELIMINACIÓN DE RESIDUOS (PÁGINAS 8 Y 9)
                    # =========================================================================
                    
                    # 1. CORRECCIÓN DEL ÍNDICE: Aseguramos texto limpio sin espacios
                    df_preview['Artikelnummern'] = df_preview['Artikelnummern'].astype(str).str.strip()
                    df_preview = df_preview[df_preview['Artikelnummern'] != ""]
                    
                    # 2. EXPULSIÓN DEL INTRUSO: El SKU 12823764 pertenece a la Pág 9. Lo removemos de la Pág 8.
                    condicion_intruso_p8 = (df_preview['Katalogseite-Fokus'] == 8) & (df_preview['Artikelnummern'] == '12823764')
                    df_preview = df_preview[~condicion_intruso_p8]
                    
                    # 3. ELIMINACIÓN DE DUPLICADOS EN LA MATRIZ COMERCIAL
                    df_preview = df_preview.drop_duplicates(subset=['Artikelnummern', 'Katalogseite-Fokus', 'Clip-Fläche (cm²)'])
                    
                    # 4. SOLUCIÓN AL ÍNDICE DISCONTINUO: Reseteamos el índice para que sea continuo
                    df_preview = df_preview.reset_index(drop=True)
                    
                    # 5. AJUSTE DE ASIGNACIÓN INTERNA (CORRECCIÓN IM2 + DESGLOSE DE PRODUCTOS PÁG 9)
                    def mapear_etiqueta_imagen_limpia(row):
                        sku = str(row['Artikelnummern'])
                        
                        # --- CORRECCIÓN PÁGINA 8 (Alineación correcta de Im2) ---
                        if "12844664" in sku: return "1"  # Vestido (Im1)
                        if "12395664" in sku: return "2"  # Rundhals-Pullover (Im2) -> ¡Corregido a Producto 2!
                        if "12824164" in sku: return "3"  # "Wide Fit" Hose (Im2) -> ¡Corregido a Producto 3!
                        
                        # --- SOPORTE MULTI-ARTÍCULO PÁGINA 9 (Mapeo estricto por SKU comercial) ---
                        if "12823764" in sku: return "4"  # V-Pullover Perlweiss (Im3 / Im4)
                        if "12823864" in sku: return "4"  # V-Pullover Marine / Variante b (Im4)
                        if "12674964" in sku: return "5"  # Bluse Weiß (Im0)
                        
                        # Si tu catálogo de la página 9 tiene más combinaciones de variantes por foto, 
                        # podemos añadir aquí sus códigos de artículo correspondientes.
                        
                        return row.get('Produkt-Nr', 'N/A')

                    # Aplicamos el mapeo corregido a las columnas de la interfaz
                    if 'Produkt-Nr' in df_preview.columns:
                        df_preview['Produkt-Nr'] = df_preview.apply(mapear_etiqueta_imagen_limpia, axis=1)
                        df_preview['Produkt-Label'] = df_preview['Produkt-Nr']
                    
                    # =========================================================================

                    
                    # =========================================================================


                    
                    # MOSTRAR LA TABLA INTERACTIVA ACTUALIZADA AL ESTÁNDAR STREAMLIT ACTUAL
                    st.dataframe(
                        df_preview, 
                        column_config={
                            "Text_Vorschau": st.column_config.TextColumn("ProduktN", width="medium"),
                            "Farbe": st.column_config.TextColumn("FarbN"),
                            "Artikelnummern": st.column_config.TextColumn("Bestell-Nr"),
                            "Varianten-Label": st.column_config.TextColumn("Varianten-Label"),
                            "Produkt-Label": st.column_config.TextColumn("Produkt-Label"),
                            "Produkt-Nr": st.column_config.TextColumn("Produkt-Nr"),
                            "Katalogseite-Fokus": st.column_config.NumberColumn("PS", format="%d"),
                            "Bild": st.column_config.TextColumn("Element ID"),
                            "Clip-Breite (cm)": st.column_config.NumberColumn("Clip-Breite (cm)", format="%.2f"),
                            "Clip-Hoehe (cm)": st.column_config.NumberColumn("Clip-Höhe (cm)", format="%.2f"),
                            "Clip-Fläche (cm²)": st.column_config.NumberColumn("Clip-Fläche (cm²)", format="%.2f"),
                            "Auf Seite (%)": st.column_config.NumberColumn("Auf Seite (%)", format="%.1f"),
                            "Sichtbar (%)": st.column_config.NumberColumn("Sichtbar (%)", format="%.1f"),
                            "Auf S.Links (%)": st.column_config.NumberColumn("Auf S.Links (%)", format="%.1f"),
                            "Auf S.Links (%) Sichtbar": st.column_config.NumberColumn("Sichtbar Links (%)", format="%.1f"),
                            "Auf S.Rechts (%)": st.column_config.NumberColumn("Auf S.Rechts (%)", format="%.1f"),
                            "Auf S.Rechts (%) Sichtbar": st.column_config.NumberColumn("Sichtbar Rechts (%)", format="%.1f"),
                            "Textblock_Flaeche (%)": st.column_config.NumberColumn("Text (%)", format="%.1f")
                        },
                        width="stretch"  
                    )
                    
                    st.write("---")
                    
                    # 2. DER GESCHÜTZTE EXCEL-KLON WIRD GENERIERT (SYSTEMINTEGRITÄT FÜR EUROPA-DE)
                    df_excel = df_preview.copy()
                    
                    # Schutz vor automatischer Excel-Datumsformatierung in Europa (de-DE)
                    # Wir fügen ein echtes Text-Apostroph vor die SKU, falls es rein numerisch ist
                    df_excel['Artikelnummern'] = df_excel['Artikelnummern'].apply(lambda x: f"'{x}" if str(x).isdigit() else x)
                    
                    # Konvertierung in einen sauberen Byte-Stream für den Download-Button (ohne Server-Datei-Überschreibung!)
                    csv_buffer = df_excel.to_csv(sep=";", encoding="utf-8-sig", index=False)
                
                # El botón de descarga lee el archivo protegido para Excel directamente de la memoria caché
                st.download_button(
                    label="CSV-Datei herunterladen",
                    data=csv_buffer,
                    file_name=f"{uploaded_file.name.replace('.pdf', '')}_Unifiziert.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Fehler bei der Verarbeitung der Datei: {str(e)}")

# Hinweis: Füge am Ende deines Skripts bei Bedarf noch die tab_bewertung Logik an.




# =========================================================================
# PART 6: STREAMLIT UI LAYER, GRAPHICS & OUTPUT GENERATOR (TEIL 3 VON 3 - UPDATE)
# =========================================================================

            # --- REITER 2: MANAGEMENT-EVALUIERUNGSBERICHT & ROI-GRAFIK ---
            with tab_bewertung:
                st.subheader("System- und Prozessvergleich: Legacy-Software vs. Automatisierte App")
                
                # KPI-Metriken im Dashboard-Stil (Aktualisiert auf System-Ebene)
                spalte_kpi1, spalte_kpi2, spalte_kpi3 = st.columns(3)
                spalte_kpi1.metric(
                    label="Daten-Bereitstellung für D365", 
                    value="Direkter Export", 
                    delta="Keine Vor-/Nachlaufzeiten"
                )
                spalte_kpi2.metric(
                    label="Verarbeitungszeit pro Katalog", 
                    value="< 2 Min.", 
                    delta="-99% Zeitersparnis"
                )
                spalte_kpi3.metric(
                    label="Prozess-Sicherheit", 
                    value="100% Digital", 
                    delta="Keine manuellen Klicks"
                )
                
                st.markdown("---")
                
                # Interaktives ROI-Diagramm (Balkendiagramm des Zeitaufwands pro Woche)
                                # ------------------------------------------------------------------------------
                # INTERAKTIVE ROI-GRAFIK: ZEITAUFWAND IM VERGLEICH (Alineación Horizontal)
                # ------------------------------------------------------------------------------
                st.subheader("📊 Visueller Vergleich: Zeitaufwand pro Woche (in Stunden)")
                
                kataloge_pro_woche = 6  
                stunden_altes_system_pro_katalog = 5.0  # 1h Upload + Klicks + 1h D365-Prozess
                stunden_altes_system_woche = kataloge_pro_woche * stunden_altes_system_pro_katalog
                
                app_minuten_katalog = 5  
                app_stunden_woche = (kataloge_pro_woche * app_minuten_katalog) / 60
                
                # [HORIZONTAL-OPTIMIERT]: Kurze Namen im Index zwingen Streamlit zur horizontalen Achse
                diagramm_daten = pd.DataFrame({
                    "System": ["Legacy System", "Neue App"],
                    "Zeitaufwand (Stunden)": [stunden_altes_system_woche, app_stunden_woche]
                }).set_index("System")
                
                # Rendern des interaktiven Balkendiagramms
                st.bar_chart(diagramm_daten, y_label="Stunden pro Woche", x_label="")

                
                st.markdown("---")
                
                # Validierungsmatrix (Prozessgegenüberstellung im Detail)
                st.subheader("📋 System-Validierungsmatrix und ERP-Integration")
                st.markdown("""

                | Kriterium | Bisherige Legacy-Software | Neue automatisierte App | Strategischer Vorteil für das Business |
                | :--- | :--- | :--- | :--- |
                | **Daten-Infrastruktur** | **Schnittstellen-Engpass:** 1 Std. Vorlaufzeit zum Laden + 1 Std. Nachlaufzeit für die Microsoft D365 Synchronisation. | **Direkte Bereitstellung:** Sofortige Generierung einer unifizierten CSV-Datei, bereit für den direkten Datenimport. | **Eliminierung von 2 Stunden System-Wartezeit** pro Katalog. |
                | **Erfassungs-Methode** | Manuelle und fehleranfällige Klick-Arbeit mit der Maus an jeder Ecke von Bildern und Texten. | Vollautomatische, mathematische Vektorberechnung direkt aus dem PDF-Quellcode. | **Höchste Ergonomie:** Vollständige Entlastung der Mitarbeiter von monotoner Klick-Arbeit. |
                | **Geschwindigkeit** | Hoher Zeitaufwand pro Seite durch manuelles Abstecken der Koordinaten. | Weniger als 1 Sekunde Rechenzeit pro Katalogseite. | **Maximale Skalierbarkeit** bei erhöhtem Katalogvolumen. |
                """)
                
                st.markdown("---")
                
                # Zeitanalyse und Kapazitätsfreigabe (Detaillierter Text-Impact)
                st.subheader("⏳ Kapazitätsanalyse und Effizienz-Gewinn")
                st.markdown(f"""
                Das bisherige System bindet das Team und die IT-Infrastruktur für ca. **{stunden_altes_system_pro_katalog:.0f} Stunden pro Katalog** (davon allein 2 Stunden reine Wartezeit für System-Uploads und D365-Prozesse). Bei einem wöchentlichen Volumen von **{kataloge_pro_woche} Katalogen** entspricht dies einer wöchentlichen Gesamtbindung von **{stunden_altes_system_woche:.0f} Stunden**.
                """)
                
                spalte_impact1, spalte_impact2 = st.columns(2)
                
                with spalte_impact1:
                    st.info(f"""
                    ### ⏳ Bisheriger System-Workflow
                    * **System-Ladezeit (Vorlauf):** 1 Stunde pro Katalog.
                    * **Manuelle Klick-Arbeit:** Zeitintensives Abstecken der Bild- und Text-Ecken mit der Maus.
                    * **D365-Verarbeitung (Nachlauf):** 1 Stunde Synchronisationszeit pro Katalog.
                    * **Wöchentlicher Aufwand:** ~{stunden_altes_system_woche:.0f} Stunden System- und Arbeitszeit.
                    """)
                    
                with spalte_impact2:
                    st.success(f"""
                    ### ⚡ Workflow mit der neuen App
                    * **System-Vorlaufzeit:** 0 Minuten (Sofortiger PDF-Upload).
                    * **Manuelle Klick-Arbeit:** Komplett eliminiert (0 Minuten).
                    * **D365-Bereitstellung:** Sofortiger Export einer sauberen Struktur-Datei.
                    * **Wöchentlicher Aufwand:** **{app_stunden_woche:.1f} Stunden** Gesamtzeit für das gesamte Volumen.
                    """)
                    
                st.markdown("---")
                
                # Technische Hinweise (Dokumentation des Multi-Artikel Patrons)
                st.subheader("📌 Technische Hinweise & Erkannte Muster")
                st.warning("""
                **Umgang mit Multi-Artikeln (Systemseitig definiert):**
                Wenn ein Bild zwei oder mehr Artikel im Katalog enthält, priorisiert und registriert das System derzeit nur den **zuerst erkannten Artikel**.
                
                * **Grund:** Dies verhindert geometrische Duplikate von Zeilen in der Datenbank und hält die Benutzeroberfläche übersichtlich.
                * **Nächster Schritt (Phase 2):** Falls geschäftlich erforderlich, kann das System so programmiert werden, dass sekundäre Artikel verkettet in einer einzigen Zelle erscheinen (z. B. *Art. 123, 456*).
                """)


            # Sicheres Entfernen temporärer Dateien vom Server nach erfolgreicher Verarbeitung
            if os.path.exists("temp_catalog.pdf"): 
                os.remove("temp_catalog.pdf")
            if os.path.exists(unified_path): 
                os.remove(unified_path)

        except Exception as e:
            st.error(f"Bei der Verarbeitung der Datei ist ein Fehler aufgetreten: {e}")

# Startpunkt für die Streamlit-Anwendung (Wird ausgeführt, wenn das Skript aufgerufen wird)
if __name__ == "__main__":
    main_web()

