import cv2
import os
import progressbar


path_in = 'records/'
path_out = ''
name = 'recording3_marked.avi'
fps = 24.0

def make_video(path_in, path_out, name):
    frames = []

    print('Scanning folder')
    bar = progressbar.ProgressBar(maxval=len(os.listdir(path_in))).start()

    files = sorted(os.listdir(path_in))
    for file in files:
        bar.update(len(frames))
        frames.append(cv2.imread('{}{}'.format(path_in, file)))

    bar.finish()

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('{}{}'.format(path_out, name), fourcc, fps, (w, h), isColor=len(frames[0].shape) > 2)

    print('Creating video')
    bar = progressbar.ProgressBar(maxval=len(frames)).start()

    for i, frame in enumerate(frames):
        bar.update(i)
        writer.write(frame)

    bar.finish()

if __name__ == '__main__':
    make_video(path_in, path_out, name)
    print('Complete')
