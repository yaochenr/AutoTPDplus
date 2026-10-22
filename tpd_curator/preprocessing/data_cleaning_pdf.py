import glob
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

TOO_LONG_TEXT = 100000


exclude_sections = [
    "ANIMAL USE",
    "ASSOCIATED CONTENT",
    "AUTHOR INFORMATION",
    "ACKNOWLEDGMENTS",
    "REFERENCES",
    "EXPERIMENTAL SECTION",
    "ABBREVIATIONS USED",
    "Letter",
    "Supplementary Material"

]

def load_md_from_path(path):
    with open(path) as f:
        data = f.read()
    return data

rm_authors_and_links = re.compile(r"\[.*?\]\((https?:\/\/.*?)\)(?:\s*field)?"), ""
rm_figure_info = re.compile(r"!\[\]\(.*?\)"), ""
rm_ref_brackets = re.compile(r"\[\.?\d+(?:,\d+)?\]\(#page-.*?\)"), ""
rm_span_tags = re.compile(r'<span\s+id="page-\d+-\d+"></span>'), ""
rm_page_info = re.compile(r"\(#page.*?\)"), ""
rm_figure_references = re.compile(r"\(Figure.*?\)"), ""
rm_table_references = re.compile(r"\(Table.*?\)"), ""
rm_received_date = re.compile(r"Received:\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}\s*"), ""
rm_figure_captions = (re.compile(r'(?is)^Figure\s+\d+\..*?(?=\n\s*\n|$)', re.MULTILINE), "")
rm_schema_captions = (re.compile(r'^#{1,6}\s*Scheme\s+\d+\..*$', re.MULTILINE), "")
rm_square_brackets_brackets = (re.compile(r'\[.*?\)'), "")

find_headers = re.compile("(#{1,6}.*)\\n")
def get_headers(md, verbose=False):

    headers = []
    for match in find_headers.finditer(md):
        span = match.span()
        headers.append((md[span[0] : span[1]], span))
    if verbose:
        for h in headers:
            print(h)
    return headers

def clean_small_stuff_in_text(md):
    rm_replace = [
        rm_authors_and_links,
        rm_figure_info,
        rm_ref_brackets,
        rm_span_tags,
        rm_figure_references,
        rm_page_info,
        rm_table_references,
        rm_received_date,
        rm_figure_captions,
        rm_schema_captions,
        rm_square_brackets_brackets
    ]    
    for reg, replacement in rm_replace:
        md = reg.sub(replacement, md)

    return md

def remove_content_before_first_header(md):
    headers = get_headers(md)
    if not headers:
        return md
    _, first_span = headers[0]
    md = md[first_span[0] :]

    return md

def remove_sections(md, sections_to_remove):
    headers = get_headers(md) 
    removal_intervals = []
    for i, (header_text, (start, end)) in enumerate(headers):
        if any(sec.upper() in header_text.upper() for sec in sections_to_remove):
            current_level = header_text.count('#')
            removal_start = start
            removal_end = len(md)
            for j in range(i + 1, len(headers)):
                next_header_text, (next_start, next_end) = headers[j]
                next_level = next_header_text.count('#')
                if next_level <= current_level:
                    removal_end = next_start
                    break
            removal_intervals.append((removal_start, removal_end))

    for start, end in sorted(removal_intervals, key=lambda x: x[0], reverse=True):
        md = md[:start] + md[end:]

    return md

def clean_md(md_path, sections_to_remove):
    md = load_md_from_path(md_path)
    md = remove_sections(md, sections_to_remove)
    md = clean_small_stuff_in_text(md)
    md = remove_content_before_first_header(md)
    md = re.sub(r"\n\s*\n", "\n", md)
    return md


def clean_pdf_data(pdf_path: Union[str, Path], output_dir: Union[str, Path],
                   sections_to_remove: Optional[List[str]] = None, 
                   save_images: bool = False) -> List[Dict[str, Any]]:

    from ..preprocessing.convert_pdf_to_md import convert_pdf_to_md
    from marker.config.parser import ConfigParser
    
    if sections_to_remove is None:
        sections_to_remove = exclude_sections
    
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF path does not exist: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting PDF(s) from {pdf_path} to markdown...")
    config_parser = ConfigParser({"output_format": "markdown"})
    convert_pdf_to_md(config_parser, str(pdf_path), str(output_dir), save_images=save_images)

    md_files = list(output_dir.rglob("*.md"))
    if not md_files:
        print(f"No markdown files found in {output_dir}")
        return []

    print(f"Found {len(md_files)} markdown files. Cleaning...")

    cleaned_data = []
    for i, md_file in enumerate(md_files, start=1):
        print(f"Cleaning: {md_file.name}")
        try:
            cleaned_content = clean_md(str(md_file), sections_to_remove)

            data_obj = {
                "file_number": i,
                "source_type": "pdf",
                "source_file": str(md_file.stem),
                "content": cleaned_content
            }
            cleaned_data.append(data_obj)
            
        except Exception as e:
            print(f"Failed to clean {md_file.name}: {e}")
            continue
    
    print(f"Successfully cleaned {len(cleaned_data)} PDF files.")
    return cleaned_data


def clean_pmc_data(pmc_path: Union[str, Path], 
                   output_dir: Union[str, Path],
                   fields_to_remove: Optional[List[str]] = None) -> int:

    if fields_to_remove is None:
        fields_to_remove = ["extraction_date", "sections", "references"]
    
    pmc_path = Path(pmc_path)
    output_dir = Path(output_dir)
    
    if not pmc_path.exists():
        raise FileNotFoundError(f"PMC path does not exist: {pmc_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = []

    if pmc_path.is_file() and pmc_path.name == "processed.json":
        processed_files = [pmc_path]
    elif pmc_path.is_dir():
        processed_files = list(pmc_path.rglob("processed.json"))
    else:
        raise ValueError(f"Invalid PMC path. Expected directory or processed.json file: {pmc_path}")
    
    if not processed_files:
        print(f"No processed.json files found in {pmc_path}")
        return 0
    
    print(f"Found {len(processed_files)} PMC processed.json files. Cleaning...")
    
    successful_count = 0
    for json_file in processed_files:
        paper_folder = json_file.parent
        paper_name = paper_folder.name
        
        print(f"Processing: {paper_name}/processed.json")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                pmc_data = json.load(f)

            cleaned_pmc = {k: v for k, v in pmc_data.items() if k not in fields_to_remove}

            output_paper_dir = output_dir / paper_name
            output_paper_dir.mkdir(parents=True, exist_ok=True)

            images_src = paper_folder / "images"
            if images_src.exists():
                images_dst = output_paper_dir / "images"
                if images_dst.exists():
                    shutil.rmtree(images_dst)
                shutil.copytree(images_src, images_dst)
                print(f"  Copied images folder to {images_dst}")

            metadata_src = paper_folder / "metadata.json"
            if metadata_src.exists():
                metadata_dst = output_paper_dir / "metadata.json"
                shutil.copy2(metadata_src, metadata_dst)
                print(f"  Copied metadata.json to {metadata_dst}")

            cleaned_file = output_paper_dir / "cleaned.json"
            with open(cleaned_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_pmc, f, ensure_ascii=False, indent=2)
            
            successful_count += 1
            print(f"  Successfully processed {paper_name}")
            
        except Exception as e:
            print(f"Failed to process {paper_name}: {e}")
            continue
    
    print(f"Successfully processed {successful_count} PMC papers in folder structure.")
    return successful_count


def clean_data_auto(input_path: Union[str, Path], output_dir: Union[str, Path],
                    data_type: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if data_type is None:
        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                data_type = 'pdf'
            elif input_path.name == 'processed.json':
                data_type = 'pmc'
            else:
                raise ValueError(f"Cannot auto-detect data type for file: {input_path}")
        elif input_path.is_dir():
            if list(input_path.rglob("processed.json")):
                data_type = 'pmc'
            elif list(input_path.rglob("*.pdf")):
                data_type = 'pdf'
            else:
                raise ValueError(f"No PDF or processed.json files found in: {input_path}")

    print(f"Auto-detected data type: {data_type}")

    if data_type == 'pdf':
        return clean_pdf_data(input_path, output_dir, **kwargs)
    elif data_type == 'pmc':
        return clean_pmc_data(input_path, **kwargs)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def save_cleaned_data(cleaned_data: List[Dict[str, Any]], output_dir: Union[str, Path],
                      file_prefix: str = "cleaned") -> List[Path]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for data in cleaned_data:
        file_number = data.get("file_number", "unknown")
        source_file = data.get("source_file", f"file_{file_number}")

        filename = f"{file_prefix}_{file_number}_{source_file}.json"
        output_file = output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        saved_files.append(output_file)
        print(f"Saved: {output_file}")
    
    return saved_files