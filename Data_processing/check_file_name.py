from os.path import join, basename, splitext, exists
from os import mkdir,remove
import glob

# path = r'.\raw_data'
def check_filename(path):
    allFiles = glob.glob(join(path, "**/*.csv"), recursive=True)
    files = [f for f in allFiles]                       # Full path name

    basenames = [splitext(basename(f))[0].split('.') for f in files]    # File name
    tag_type = [name[1].split('_')[0] for name in basenames]

    tags = [basenames[i][0]+'.'+tag_type[i] for i in range(len(basenames))]
    filelist = list(set(tags))

    if exists("./filelist"):
        if exists("./filelist.txt"):
            remove("./filelist.txt")
    else:
        mkdir("./filelist")

    with open("./filelist/filelist.txt", 'w') as f:
        f.write(str(filelist))
        f.write("\n\n")
        f.write("The number of file : %d" %len(filelist))

    return filelist

if __name__ == '__main__':
    check_filename(r'.\raw_data')