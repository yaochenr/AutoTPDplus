

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from datetime import datetime


def clean_processed_json_data(data: Dict[str, Any], fields_to_remove: List[str],
                             sections_to_remove: Optional[List[str]] = None,
                             figure_fields_to_remove: Optional[List[str]] = None) -> Dict[str, Any]:

    cleaned_data = data.copy()

    for field in fields_to_remove:
        cleaned_data.pop(field, None)

    if sections_to_remove and "sections" in cleaned_data and isinstance(cleaned_data["sections"], dict):
        sections_dict = cleaned_data["sections"]
        sections_to_remove_lower = [section.lower() for section in sections_to_remove]

        sections_to_delete = []
        for section_key in sections_dict.keys():
            if section_key.lower() in sections_to_remove_lower:
                sections_to_delete.append(section_key)

        for section_key in sections_to_delete:
            sections_dict.pop(section_key, None)
            print(f"  Removed section: {section_key}")

    if figure_fields_to_remove and "figures" in cleaned_data and isinstance(cleaned_data["figures"], list):
        figures_list = cleaned_data["figures"]
        removed_fields = set()

        for figure in figures_list:
            if isinstance(figure, dict):
                for field_name in figure_fields_to_remove:
                    if field_name in figure:
                        figure.pop(field_name, None)
                        removed_fields.add(field_name)

        if removed_fields:
            print(f"  Removed figure fields: {', '.join(sorted(removed_fields))}")

    return cleaned_data


def clean_single_json_file(input_file: Union[str, Path],
                          output_file: Optional[Union[str, Path]] = None,
                          fields_to_remove: Optional[List[str]] = None,
                          sections_to_remove: Optional[List[str]] = None,
                          figure_fields_to_remove: Optional[List[str]] = None) -> bool:

    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}")
        return False

    if not fields_to_remove:
        raise ValueError("fields_to_remove must be provided and cannot be empty")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        cleaned_data = clean_processed_json_data(data, fields_to_remove, sections_to_remove, figure_fields_to_remove)

        if output_file is None:
            output_path = input_path
        else:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

        print(f"Successfully cleaned: {input_path}")
        return True

    except Exception as e:
        print(f"Failed to clean {input_path}: {e}")
        return False


def clean_json_directory(input_dir: Union[str, Path],
                        output_dir: Optional[Union[str, Path]] = None,
                        filename: str = "processed.json",
                        fields_to_remove: Optional[List[str]] = None,
                        sections_to_remove: Optional[List[str]] = None,
                        figure_fields_to_remove: Optional[List[str]] = None) -> int:

    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Input directory does not exist: {input_path}")
        return 0

    if not fields_to_remove:
        raise ValueError("fields_to_remove must be provided and cannot be empty")

    json_files = list(input_path.rglob(filename))

    if not json_files:
        print(f"No {filename} files found in {input_path}")
        return 0

    print(f"Found {len(json_files)} files to clean...")

    successful_count = 0

    for json_file in json_files:
        if output_dir is None:
            output_file = None
        else:
            relative_path = json_file.relative_to(input_path)
            output_file = Path(output_dir) / relative_path

        if clean_single_json_file(json_file, output_file, fields_to_remove, sections_to_remove, figure_fields_to_remove):
            successful_count += 1

    print(f"Successfully cleaned {successful_count}/{len(json_files)} files")
    return successful_count


def clean_json_data(input_path: Union[str, Path],
                   output_dir: Union[str, Path],
                   fields_to_remove: Optional[List[str]] = None,
                   sections_to_remove: Optional[List[str]] = None,
                   figure_fields_to_remove: Optional[List[str]] = None,
                   paper_filter_file: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:

    if not fields_to_remove:
        raise ValueError("fields_to_remove must be provided and cannot be empty. Please specify json_fields_to_remove in your config file.")
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    allowed_papers = None
    if paper_filter_file:
        filter_path = Path(paper_filter_file)
        if not filter_path.exists():
            raise FileNotFoundError(f"Paper filter file does not exist: {filter_path}")

        try:
            with open(filter_path, 'r', encoding='utf-8') as f:
                filter_data = json.load(f)

            allowed_pmc_ids = set()
            for paper in filter_data:
                if 'pmc_id' in paper and paper['pmc_id']:
                    pmc_id = paper['pmc_id'].strip()
                    allowed_pmc_ids.add(pmc_id.lower())
                    print(f"  Added to filter: {pmc_id}")

            print(f"Loaded paper filter with {len(allowed_pmc_ids)} PMC IDs specified")
            print(f"Filter file: {filter_path}")
            allowed_papers = allowed_pmc_ids

        except Exception as e:
            raise ValueError(f"Failed to load paper filter file {filter_path}: {e}")

    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    timestamped_dir = output_dir / "cleaned_papers" / timestamp
    cleaned_output_dir = timestamped_dir / "cleaned_jsons"
    cleaned_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created timestamped directory:")
    print(f"  Cleaned JSONs: {cleaned_output_dir}")

    if input_path.is_file():
        json_files = [input_path]
    else:
        json_files = list(input_path.rglob("processed.json"))
    
    if not json_files:
        print(f"No processed.json files found in {input_path}")
        return []
    
    print(f"Found {len(json_files)} JSON files. Cleaning...")
    
    cleaned_data = []
    skipped_count = 0
    
    for i, json_file in enumerate(json_files, start=1):
        print(f"Processing: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            xml_metadata = data.get("xml_metadata", {})
            pmid = xml_metadata.get("pmid", f"PMC{i:03d}")

            if allowed_papers is not None:
                folder_name = json_file.parent.name
                if folder_name.lower() in allowed_papers:
                    print(f"  ✓ Paper matched by folder name: {folder_name}")
                else:
                    print(f"  ✗ Folder not in filter list, skipping: {folder_name}")
                    skipped_count += 1
                    continue

            folder_name = json_file.parent.name
            if folder_name != "." and folder_name.startswith('PMC'):
                file_prefix = folder_name
            elif pmid and str(pmid).startswith('PMC'):
                file_prefix = str(pmid)
            elif pmid:
                file_prefix = f"PMC{pmid}"
            else:
                file_prefix = f"PMC{i:03d}_{folder_name}"

            cleaned_json = clean_processed_json_data(data, fields_to_remove, sections_to_remove, figure_fields_to_remove)

            cleaned_filename = f"{file_prefix}_cleaned.json"
            cleaned_file_path = cleaned_output_dir / cleaned_filename

            with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_json, f, ensure_ascii=False, indent=2)

            print(f"Saved cleaned file: {cleaned_file_path}")

            data_obj = {
                "file_number": i,
                "source_type": "json",
                "source_file": file_prefix,
                "folder_name": folder_name,
                "pmid": pmid,
                "content": cleaned_json
            }
            cleaned_data.append(data_obj)
            
        except Exception as e:
            print(f"Failed to clean {json_file.name}: {e}")
            continue
    
    print(f"Successfully cleaned {len(cleaned_data)} JSON files.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files (not in filter list)")
    print(f"Cleaned files saved to: {cleaned_output_dir}")
    
    return cleaned_data, str(cleaned_output_dir)


def save_cleaned_json_data(cleaned_data: List[Dict[str, Any]], 
                          output_dir: Union[str, Path],
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
