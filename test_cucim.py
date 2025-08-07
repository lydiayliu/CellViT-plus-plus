from cucim import CuImage
import subprocess
import time
import os

from pathlib import Path
import requests
from tqdm import tqdm
import pyvips

def file_exists(directory_path: Path, file_name: str) -> bool:
    """Check if a file exists in a specific directory.

    Args:
        directory_path (Path): The path of the directory to check.
        file_name (str): The name of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    file_path = directory_path / file_name
    return file_path.exists()


def download_file(download_link: str, file_path: Path) -> None:
    """Download a file from a link and save it to a specific path.

    Args:
        download_link (str): The link to download the file from.
        file_path (Path): The path to save the downloaded file to.

    Raises:
        HTTPError: If the download request fails.
    """
    response = requests.get(download_link, stream=True)

    # Ensure the request was successful
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KiloByte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def check_and_download(
    directory_path: Path, file_name: str, download_link: str
) -> None:
    """Check if a file exists, and download it if it does not exist.

    Args:
        directory_path (Path): The path of the directory to check.
        file_name (str): The name of the file to check.
        download_link (str): The link to download the file from if it does not exist.
    """
    if not file_exists(directory_path, file_name):
        file_path = directory_path / file_name
        print("Downloading file...")
        download_file(download_link, file_path)
        print(
            f"The file {file_name} has been successfully downloaded and is located in {directory_path}."
        )
    else:
        print(f"The file {file_name} already exists in {directory_path}.")


def convert_pyramid(base_path: Path, inname: str, outname: str) -> None:
    print("Converting to pyramid image")
    image = pyvips.Image.new_from_file(base_path / inname, access="sequential")
    image.tiffsave(base_path / outname, tile=True, pyramid=True)


def check_test_database() -> None:
    """Check if the test database exists, and download it if it does not exist."""
    print("Checking Test Database")
    base_path = Path(__file__).parent.parent.parent / "test_database" / "MIDOG"
    check_and_download(
        base_path,
        "001.tiff",
        "https://springernature.figshare.com/ndownloader/files/40282099",
    )
    convert_pyramid(base_path, "001.tiff", "001_pyramid.tiff")
    base_path = Path(__file__).parent.parent.parent / "test_database" / "x20_svs"
    check_and_download(
        base_path,
        "CMU-1-Small-Region.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs",
    )
    base_path = Path(__file__).parent.parent.parent / "test_database" / "x40_svs"
    check_and_download(
        base_path,
        "JP2K-33003-2.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/JP2K-33003-2.svs",
    )
    print("Test Database is now cached on local machine.")

def log_message(message, level="INFO"):
    """Log messages with timestamps and severity levels."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def run_ray_test():
    """Run the ray_test.py script."""
    try:
        log_message("Executing 'cellvit/inference/ray_test.py'...", "INFO")
        result = subprocess.run(
            ["python", "cellvit/inference/ray_test.py"],
            check=True,
            capture_output=True,
            text=True
        )
        log_message(f"Script output:\n{result.stdout}", "SUCCESS")
    except subprocess.CalledProcessError as e:
        log_message(f"Error while running 'ray_test.py': {e.stderr}", "ERROR")
        raise e
    except Exception as e:
        log_message(f"Unexpected error while running 'ray_test.py': {e}", "ERROR")
        raise e

log_message("Checking CuCIM availability...")
log_message("Downloading example files...")
check_test_database() # potential error
log_message("Opening example Image with CuCIM")

# print working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# potential error due to lack of full path
image = CuImage("/camp/lab/swantonc/working/liul/repo/CellViT-plus-plus/test_database/x40_svs/JP2K-33003-2.svs")
print(image.size())
print(image.resolutions)
image.read_region((0, 0), (1000, 1000))
log_message("Imported CuCIM and loaded example WSI", "SUCCESS")

print("Launching Ray test...")
run_ray_test()

log_message("")
log_message("")
log_message(f"{60*'*'}")
log_message("")
log_message("")
log_message("Everything checked", "SUCCESS")
log_message("")
log_message("")
log_message(f"{60*'*'}")
