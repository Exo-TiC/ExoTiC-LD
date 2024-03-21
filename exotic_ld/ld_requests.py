import os
import requests
from tqdm import tqdm
from requests.exceptions import HTTPError, ConnectionError


def download(url, local_file_name, verbose, chunk_size=1024):
    local_dir = os.path.dirname(local_file_name)
    os.makedirs(local_dir, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        desc = "Downloading {}".format(url)
        bool_progress_bar = verbose == 0
        with tqdm(desc=desc, total=total, unit="iB", unit_scale=True,
                  unit_divisor=1024, disable=bool_progress_bar) as bar:
            with open(local_file_name, "wb") as file:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)

    except HTTPError as err:
        raise HTTPError("HTTP error occurred: url={}, msg={}"
                        .format(err.request.url, err))

    except requests.exceptions.ConnectionError as err:
        raise ConnectionError("Connection error occurred: url={}, msg={}"
                              .format(err.request.url, "Cannot connect to URL."))

    except Exception as err:
        raise err
