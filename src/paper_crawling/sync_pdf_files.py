import os
import shutil


def main():
    remote_dir = "/mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/paper/pdf"

    done_markdowns = [f.split(".")[0] for f in os.listdir("data/paper/md")]

    for file in os.listdir(remote_dir):
        if file.split(".")[0] in done_markdowns:
            print(f"Remove {os.path.join(remote_dir, file)}")
            os.remove(os.path.join(remote_dir, file))


    for file in os.listdir("data/paper/pdf"):
        if file.split(".")[0] in done_markdowns:
            continue

        if os.path.exists(os.path.join(remote_dir, file)):
            continue

        print(f"Copy {os.path.join('data/paper/pdf', file)}")
        shutil.copy(os.path.join("data/paper/pdf", file), os.path.join(remote_dir, file))



if __name__ == '__main__':
    main()
