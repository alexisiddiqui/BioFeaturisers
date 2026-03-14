# /// script
# dependencies = [
#   "markitdown[pdf]",
# ]
# ///

import re
from markitdown import MarkItDown

pdf_path = "/Users/alexi/Documents/BioFeaturisers/biofeaturisers/references/SAXS/waasm_aca95.pdf"

Z_DICT = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26
}

def clean_token(t):
    t = t.replace("~", "5").replace("$", "5").replace("q", "9")
    t = t.replace(",", ".").replace("\"", ".").replace("'", ".")
    t = t.replace("I", "1").replace("l", "1").replace("i", "1")
    t = t.replace("j", "").replace("!", "1").replace("%", "7")
    t = t.replace("..", ".").replace(". .", ".")
    
    # Surgical fix for double dots within a number
    if t.count('.') > 1:
        # If it looks like 1.23456.789123
        t = re.sub(r"(\d+\.\d{6})\.(\d+)", r"\1 \2", t)
        # fallback
        if t.count('.') > 1:
            t = re.sub(r"(\d+\.\d+)\.(\d+)", r"\1\2", t)

    # Fix concatenated floats: e.g. 0.262416120.363024
    t = re.sub(r"(\d+\.\d{5,})(\d+)", r"\1 \2", t)
    
    # Specific fix for Mg/Na/Li noise:
    if "3539403" in t: t = t.replace("3539403", "3.539403")
    
    return t

def extract_floats(line):
    # Extract ALL tokens that could be numbers after cleaning
    parts = line.split()
    nums = []
    for p in parts:
        cleaned = clean_token(p)
        # Split by space if clean_token inserted any
        for sub in cleaned.split():
            # Find floats
            found = re.findall(r"[-+]?\d*\.?\d+", sub)
            for val_str in found:
                if '.' in val_str:
                    try:
                        val = float(val_str)
                        if abs(val) < 3000:
                            nums.append(val)
                    except ValueError:
                        pass
                elif len(val_str) >= 4: # Flattened float?
                    try:
                        val = float(val_str[0] + "." + val_str[1:])
                        nums.append(val)
                    except ValueError:
                        pass
    return nums

def parse_waasm_pdf(pdf_path, target_elements):
    md = MarkItDown()
    result = md.convert(pdf_path)
    text = result.text_content
    
    name_map = {
        "He": "He", "IA": "Li", "1A": "Li", "Li": "Li", "Be": "Be", "B ": "B", 
        "C ": "C", "N ": "N", "O ": "O", "F ": "F", "Ne": "Ne", "Ns": "Na", "Na": "Na",
        "M I": "Mg", "MI": "Mg", "Mg": "Mg", "A/": "Al", "A1": "Al", "Al": "Al", 
        "Sl": "Si", "S1": "Si", "Si": "Si", "P ": "P", "S ": "S", 
        "CI": "Cl", "Cl": "Cl", "As\"": "Ar", "As": "Ar", "Ar": "Ar", "K ": "K", "Cs": "Ca", "Ca": "Ca",
        "Sc": "Sc", "Ti": "Ti", "V ": "V", "Cr": "Cr", "Mn": "Mn", "Fe": "Fe",
        "H": "H"
    }

    element_data = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        matched_elem = None
        for k, v in name_map.items():
            if line.startswith(k):
                remainder = line[len(k):].strip()
                if re.match(r"^[0-9]*[+-]", remainder): continue
                matched_elem = v
                break
        
        if not matched_elem: continue
        
        nums = extract_floats(line)
        if len(nums) >= 11:
            # We want a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 c
            # Try all windows of 11 if many numbers
            best_vals = None
            min_diff = 100
            Z = Z_DICT[matched_elem]
            
            for i in range(len(nums) - 10):
                window = nums[i:i+11]
                a_sum = sum(window[0:10:2])
                total = a_sum + window[10]
                diff = abs(total - Z)
                if diff < min_diff:
                    min_diff = diff
                    best_vals = window
            
            if best_vals and min_diff < 1.5:
                element_data[matched_elem] = best_vals

    return element_data

target_elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe"]

print(f"Extracting Neutral Atoms from {pdf_path} (H to Fe)...")
data = parse_waasm_pdf(pdf_path, target_elements)

print("\n_WK_COEFFS: dict[str, _WKEntry] = {")
for elem in target_elements:
    if elem in data:
        val = data[elem]
        a = (val[0], val[2], val[4], val[6], val[8])
        b = (val[1], val[3], val[5], val[7], val[9])
        c = val[10]
        print(f"    \"{elem}\": _WKEntry(")
        print(f"        a={tuple(round(x, 6) for x in a)},")
        print(f"        b={tuple(round(x, 6) for x in b)},")
        print(f"        c={round(c, 6)},")
        print(f"    ),")
    else:
        print(f"    # {elem} NOT EXTRACTED")
print("}")
