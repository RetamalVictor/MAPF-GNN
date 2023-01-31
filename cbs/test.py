import yaml

with open("cbs\input.yaml", 'r') as param_file:
    try:
        param = yaml.load(param_file, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)

print(param)