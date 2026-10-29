"""Generate a summary of image download status for a mined paper directory."""

import json
from pathlib import Path

from tpd_curator.pmc_miner.core.storage import StorageManager


def print_image_summary(storage: StorageManager) -> None:
    all_papers = storage.get_all_paper_ids()
    papers_with_images = 0
    total_images = 0

    print("PMC Image Download Summary")
    print("=" * 50)

    for pmcid in all_papers:
        paper_dir = storage.get_paper_dir(pmcid)
        images_dir = paper_dir / "images"

        if not images_dir.exists():
            continue

        papers_with_images += 1
        image_files = (
            list(images_dir.glob("*.jpg"))
            + list(images_dir.glob("*.png"))
            + list(images_dir.glob("*.gif"))
        )
        paper_image_count = len(image_files)
        total_images += paper_image_count

        metadata_file = images_dir / "figures_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            with_path = len([fig for fig in metadata if "local_image_path" in fig])
            print(f"  {pmcid}: {paper_image_count} images, {with_path} with metadata")
        else:
            print(f"  {pmcid}: {paper_image_count} images (no metadata)")

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Total papers: {len(all_papers)}")
    print(f"  Papers with images: {papers_with_images}")
    print(f"  Total images downloaded: {total_images}")
    if all_papers:
        print(f"  Coverage: {papers_with_images / len(all_papers) * 100:.1f}%")
    if papers_with_images:
        print(f"  Average images per paper: {total_images / papers_with_images:.1f}")

    missing = [p for p in all_papers if not (storage.get_paper_dir(p) / "images").exists()]
    if missing:
        print(f"\nPapers without images ({len(missing)}):")
        for pmcid in missing:
            print(f"  - {pmcid}")


def main(data_dir: str) -> None:
    storage = StorageManager(data_dir=Path(data_dir))
    print_image_summary(storage)
