import io
import json, pickle

def write_json_data(path, data):
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def dump_object(obj, path):
    pickle.dump(obj, open(path, 'wb'))

def get_method_key(key, method):
        return "{}_{}".format(method, key)
