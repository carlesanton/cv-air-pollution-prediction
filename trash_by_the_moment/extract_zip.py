import argparse
import gzip
import sys

import zipfile36 as zipfile


def extract_zip_file(file_obj, location):
    """
    Extracts a bunch of images from a zip file.
    """
    # print '    extracting zip...',
    sys.stdout.flush()

    zf = zipfile.ZipFile(file_obj, "r")

    print(zipfile.is_zipfile(file_obj))
    zf.extractall()
    zf.close()
    file_obj.close()

    # print 'done.'
    sys.stdout.flush()


def ffff(o, d):
    with zipfile.GzipFile(o) as Zip:
        for ZipMember in Zip.infolist():
            Zip.extract(ZipMember, path=d)


def unzip_file_to(zip_path, destiny_path, delete=0):
    try:
        zip_ref = zipfile.ZipFile(zip_path, "r")
        zip_ref.extractall(destiny_path)
        zip_ref.close()
        if delete:
            os.remove(zip_path)
        return 1
    except zipfile.BadZipFile:
        print("BadZipFile exception with folder: " + zip_path)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_origin_path", help="specify the path containing the zip file", type=str
    )
    parser.add_argument(
        "file_destiny_path",
        help="specify the path where the zip file must be extracted)",
        type=str,
    )

    args = parser.parse_args()
    origin = args.file_origin_path
    destiny = args.file_destiny_path
    print(origin)
    print(destiny)
    ffff(origin, destiny)
