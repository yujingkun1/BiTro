import os
import logging
from histolab.slide import Slide
from histolab.tiler import GridTiler
import histolab.filters.image_filters as imf
import histolab.filters.morphological_filters as mof
from histolab.masks import BiggestTissueBoxMask
import openslide

# Logging setup
logging.basicConfig(filename="processing.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Define input and output paths
IMAGE_FOLDER = "./demo_data/WSI"
OUTPUT_PATCHES_FOLDER = "./demo_data/output_patches"  # Separate output folder

# Create output folder
os.makedirs(OUTPUT_PATCHES_FOLDER, exist_ok=True)

# Configure GridTiler
grid_tiles_extractor = GridTiler(
    tile_size=(512, 512),  
    level=0,              
    check_tissue=True,    
    pixel_overlap=0,      
    prefix="",            
    suffix=".png"        
)

# Define extraction mask
extraction_mask = BiggestTissueBoxMask()
extraction_mask.custom_filters = [
    imf.BluePenFilter(), imf.RgbToGrayscale(), imf.OtsuThreshold(),
    mof.BinaryDilation(), mof.RemoveSmallHoles(), mof.RemoveSmallObjects()
]

# Load processed WSI records
processed_wsi = set()
if os.path.exists("processed_wsi.txt"):
    with open("processed_wsi.txt", "r") as f:
        processed_wsi = set(f.read().splitlines())

# Iterate through WSI files
for filename in os.listdir(IMAGE_FOLDER):
    if not filename.endswith(".svs"):
        continue

    wsi_name = os.path.splitext(filename)[0]
    slide_output_folder = os.path.join(OUTPUT_PATCHES_FOLDER, wsi_name)

    # **Check if fully processed**
    if wsi_name in processed_wsi:
        print(f"Skipping {filename}, already processed (found in processed_wsi.txt).")
        continue

    # **Check if target folder already contains patches**
    existing_patches = 0
    if os.path.exists(slide_output_folder):
        existing_patches = len([f for f in os.listdir(slide_output_folder) if f.endswith(".png")])
        if existing_patches > 0:
            print(f"Skipping {filename}, already contains {existing_patches} patches.")
            continue

    try:
        image_path = os.path.join(IMAGE_FOLDER, filename)
        os.makedirs(slide_output_folder, exist_ok=True)

        slide = Slide(image_path, processed_path=slide_output_folder)
        slide_os = openslide.OpenSlide(image_path)
        magnification = slide_os.properties.get('openslide.objective-power', 'Unknown')

        print(f"\n--- Processing {filename} ---")
        print(f"Slide name: {slide.name}")
        print(f"Magnification: {magnification}")

        tile_prefix = f"{wsi_name}_patch_"
        grid_tiles_extractor.prefix = tile_prefix
        print(f"Setting tile prefix to: {tile_prefix}")

        # Extract and save patches
        print(f"Extracting tiles for {filename}...")
        grid_tiles_extractor.extract(slide, extraction_mask=extraction_mask)

        # Count patches
        patch_count = len([f for f in os.listdir(slide_output_folder) if f.endswith(".png")])
        print(f"Finished extracting tiles for {filename}. Total patches: {patch_count}")

        # Record processed WSI
        with open("processed_wsi.txt", "a") as f:
            f.write(wsi_name + "\n")

        logging.info(f"Processed {filename} successfully. Total patches: {patch_count}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        logging.error(f"Failed to process {filename}: {e}")
        continue  # Continue to the next file

print("\nAll slides processed successfully!")
