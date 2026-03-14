import os
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration for the download
BASE_URL = "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_events/v2.0.1/edf/"
DOWNLOAD_DIR = "./edf"
IGNORE_EXTENSIONS = ('.html', '.htm', '.shtml')
USERNAME = "nedc-tuh-eeg"
PASSWORD = "RLYF8ZhBMZwNnsYA8FsP"

# Create the download directory if it doesn't exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_filename_from_url(url):
    """
    Get filename from URL path
    """
    path = urlparse(url).path
    return os.path.basename(path)

def file_exists_and_complete(filepath, expected_size):
    """Check if the file exists and its size matches the expected size."""
    return os.path.exists(filepath) and os.path.getsize(filepath) == expected_size

def scrape_links(url, base_url=BASE_URL, visited=None, session=None):
    if visited is None:
        visited = set()
    if session is None:
        session = requests.Session()
        session.auth = (USERNAME, PASSWORD)

    response = session.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(base_url, href)
        if is_valid(full_url) and not any(full_url.endswith(ext) for ext in IGNORE_EXTENSIONS):
            # Add link only if it does not end with an ignored extension
            links.append(full_url)
            if full_url not in visited:
                visited.add(full_url)
                # Recurse into directories that are not ignored extensions
                links.extend(scrape_links(full_url, base_url, visited, session))
    return links

def download_file(url, session, dir=DOWNLOAD_DIR, file_progress=None):
    local_filename = os.path.join(dir, get_filename_from_url(url))
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))

        # Check if file already exists and is complete
        if file_exists_and_complete(local_filename, total_size_in_bytes):
            print(f"File already exists and is complete: {local_filename}")
            if file_progress:
                file_progress.update(1)
            return

        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, leave=False)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        if file_progress:
            file_progress.update(1)

def main():
    start_url = BASE_URL
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    # Collect all file links before downloading
    all_links = list(set(scrape_links(start_url, session=session)))
    print(f"Total files to download: {len(all_links)}")

    # Download files with a global progress bar
    with tqdm(total=len(all_links), desc="Overall Progress", position=0) as file_progress:
        for link in all_links:
            with tqdm(desc=get_filename_from_url(link), position=1, leave=False) as file_pbar:
                try:
                    download_file(link, session, file_progress=file_progress)
                    file_progress.set_description(f"Downloaded {get_filename_from_url(link)}")
                except Exception as e:
                    print(f"Failed to download {link}: {e}")
                finally:
                    file_progress.update(1)


main()
