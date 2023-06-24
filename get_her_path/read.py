def read_id2path_tuple_file(file_paths):
    id2path = {}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            print('loading a path file...   ' + file_path)
            for line in f:
                th = line.strip('\n').split('\t')
                print(th)
                id2path[int(th[0])] = (th[1], th[2])
    return id2path

id2path = read_id2path_tuple_file(["./paths_1", "./paths_2"])
print(id2path[0][0])