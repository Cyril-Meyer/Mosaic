import sys
import os
import numpy as np
from PIL import Image
import skimage as ski
import skimage.metrics
from numba import jit
import distancemap as dm


class Progress:
    def __init__(self, total):
        self.total = total
        self.progress = 0

    def update(self):
        self.progress += 1
        sys.stdout.write("Progress: %f%% %s" % (100 * self.progress / self.total, "\r"))
        sys.stdout.flush()


def metric_mse(x, y):
    return skimage.metrics.mean_squared_error(x, y)


def get_mosaic(source_filename,
               output_filename,
               tiles_folder,
               destination_size=(4096, 4096),
               tile_size=(64, 64),
               tile_max_usage=1,
               tile_max_number=float('inf'),
               metric=metric_mse,
               color_mode='RGB'):

    # ---------------------------------------------------------------------------------------------------------------- #
    # Arguments checks
    if not (destination_size[0] % tile_size[0] == 0 and destination_size[1] % tile_size[1] == 0):
        print("ERROR: destination size and tile size are not multiple")
        exit(1)

    channels = 0
    if color_mode == 'RGB':
        channels = 3
    elif color_mode == 'L':
        channels = 1
    else:
        print("ERROR: invalid color mode")
        exit(1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Read and prepare
    source = Image.open(source_filename).convert(color_mode)
    print("source size :", source.size)
    destination = source.resize(destination_size)
    print("output size :", destination.size)

    tiles_files = [f for f in os.listdir(tiles_folder)
                   if (os.path.isfile(os.path.join(tiles_folder, f)) and ".png" in f)]

    x_tiles = destination_size[0] // tile_size[0]
    y_tiles = destination_size[1] // tile_size[1]
    print("tiles needed :", x_tiles * y_tiles)
    if x_tiles * y_tiles > len(tiles_files) * tile_max_usage:
        print("ERROR: not enough tiles, tiles available :", len(tiles_files) * tile_max_usage)
        exit(1)
    print("tiles available :", len(tiles_files))
    print("tiles with usage :", len(tiles_files) * tile_max_usage)

    if len(tiles_files) > tile_max_number:
        tiles_files = tiles_files[0:tile_max_number]
        print("tiles selected :", len(tiles_files))

    tileset = np.zeros(((len(tiles_files),) + tile_size + (channels,)), dtype=np.uint8)
    tileset_usage = np.full(len(tiles_files), tile_max_usage)
    progress_status = Progress(len(tiles_files))
    for f in range(len(tiles_files)):
        tileset[f] = np.array(Image.open(os.path.join(tiles_folder, tiles_files[f])).resize(tile_size).convert(color_mode))
        progress_status.update()

    destination = np.array(destination)
    print("working destination array shape :", destination.shape, destination.dtype)
    print("working tileset array shape :", tileset.shape, tileset.dtype)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute Mosaic
    x_pos = (x_tiles//2)
    y_pos = (y_tiles//2)

    distmap = np.zeros((x_tiles, y_tiles), dtype=np.bool)

    x_pos_2 = x_pos+1
    y_pos_2 = y_pos+1

    if x_tiles%2 == 0:
        x_pos -= 1
    if y_tiles%2 == 0:
        y_pos -= 1

    distmap[x_pos:x_pos_2, y_pos:y_pos_2] = True
    distmap = dm.distance_map_from_binary_matrix(distmap)

    dist = 0

    progress_status = Progress(x_tiles*y_tiles)
    while True:
        batch = np.argwhere(distmap == dist)
        if batch.size == 0:
            break
        dist += 1
        np.random.shuffle(batch)

        for p in batch:
            patch = destination[p[0]*tile_size[0]:(p[0]+1)*tile_size[0], p[1]*tile_size[1]:(p[1]+1)*tile_size[1]]

            best = float('inf')
            bestid = -1

            for tileid in range(tileset.shape[0]):
                score = metric(patch, tileset[tileid])
                if score < best:
                    best = score
                    bestid = tileid

            destination[p[0] * tile_size[0]:(p[0] + 1) * tile_size[0], p[1] * tile_size[1]:(p[1] + 1) * tile_size[1]] = tileset[bestid]
            tileset_usage[bestid] = tileset_usage[bestid]-1
            if tileset_usage[bestid] == 0:
                tileset = np.delete(tileset, bestid, 0)
                tileset_usage = np.delete(tileset_usage, bestid, 0)

            progress_status.update()
    print()

    # ---------------------------------------------------------------------------------------------------------------- #
    # Save result
    print("saving output")
    destination = Image.fromarray(destination)
    destination.save(output_filename)


get_mosaic("source.png",
           "destination.png",
           "../out_200k",
           destination_size=(4096, 4096),
           tile_size=(32, 32),
           tile_max_usage=1,
           tile_max_number=float('inf'),
           metric=metric_mse,
           color_mode='RGB')

