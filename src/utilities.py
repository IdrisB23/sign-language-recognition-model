import json
import pathlib

def save_dict_as_JSON_file(path_: pathlib.Path, data_: dict):
    if not path_.exists():
        path_.touch()
    with open(path_, 'w') as f:
        json.dump(data_, f, indent=2)
        print(f'Saved dictionary to the {path_.resolve()} JSON file.')
        f.close()


def read_dict_from_JSON(path_: pathlib.Path):
    with open(path_, 'r') as f:
        result: dict = json.load(f)
        f.close()
    return result

def create_dummy_JSON_file(path_: pathlib.Path):
    if not path_.exists():
        path_.touch()
    data = dict()
    save_dict_as_JSON_file(path_, data)