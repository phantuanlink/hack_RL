import imageio.v3 as iio, sys
f = iio.imread(sys.argv[1], index=None)
print(f.shape)
iio.imwrite(sys.argv[2], f[len(f)//2])
