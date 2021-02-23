def to_img(x):
    if x.shape[2] != 28 or x.shape[3] != 28:
        x = x.view(x.shape[0], 1, 28, 28)
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x
