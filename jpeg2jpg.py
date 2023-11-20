import os
import argparse

def rename_jpeg_to_jpg(directory):
    """ 指定されたディレクトリおよびそのサブディレクトリ内のすべての .JPEG ファイルを .jpg に変更 """
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith('.jpeg'):
                original_path = os.path.join(dirpath, filename)
                new_filename = filename[:-5] + '.jpg'
                new_path = os.path.join(dirpath, new_filename)
                os.rename(original_path, new_path)
                print(f'Changed file name: {original_path} -> {new_path}')

def main():
    parser = argparse.ArgumentParser(description='Convert .JPEG files to .jpg')
    parser.add_argument('directory', type=str, help='Directory to search for .JPEG files')
    args = parser.parse_args()
    rename_jpeg_to_jpg(args.directory)

if __name__ == '__main__':
    main()
