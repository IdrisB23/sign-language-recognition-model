import json
from pathlib import Path

def save_dict_as_JSON_file(path_: Path, data_: dict):
    if not path_.exists():
        path_.touch()
    with open(path_, 'w') as f:
        json.dump(data_, f, indent=2)
        print(f'Saved dictionary to the {path_.resolve()} JSON file.')
        f.close()


def read_dict_from_JSON(path_: Path):
    with open(path_, 'r') as f:
        result: dict = json.load(f)
        f.close()
    return result

def create_dummy_JSON_file(path_: Path):
    if not path_.exists():
        path_.touch()
    data = dict()
    save_dict_as_JSON_file(path_, data)


def get_gloss_2_nb_of_vids(path_: Path):
    gloss_2_nb_orig_vids = dict()
    for class_p in path_.iterdir():
        original_vids = class_p.glob('*')
        original_vids = [vid for vid in original_vids if 'augmented' not in vid.name]
        gloss_2_nb_orig_vids[class_p.name] = len(original_vids)
    return gloss_2_nb_orig_vids


if __name__ == '__main__':
    gloss_2_nb_orig_vids = get_gloss_2_nb_of_vids(Path('data/videos/train/cropped'))
    gloss_2_nb_orig_vids_sorted = sorted(gloss_2_nb_orig_vids.items(), key=lambda x: x[1])
    save_dict_as_JSON_file(Path('data/assets/gloss_2_nb_orig_vids_sorted_by_nb.json'), gloss_2_nb_orig_vids_sorted)