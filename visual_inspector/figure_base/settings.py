"""global data structure"""
from colour import Color


gen2sorted_indices = {} # Global sorted index for sorting BCs
fitness_plot = None
cloud_plots = set()
canvas2cloud_plot = {} # Figure canvas to figure object

noise = None # Global Noise Table
numBins = 5 # Number of Color Bins for ColorBar
assert numBins > 1

COLORS = [
    (Color('#f9d9d9'), Color('#d61515')), # red
    (Color('#d9ddfb'), Color('#0b1667')), # blue
    (Color('#9aecb8'), Color('#045c24')), # green
    (Color('#ffbef9'), Color('#ce00bb')), # pink
    (Color('#d0d0d0'), Color('#000000')), # black
    (Color('#f2d6b9'), Color('#996633')), # brown
    (Color('#d5b2ec'), Color('#9900FF')), # purple
    (Color('#baffff'), Color('#009999')), # teel
    (Color('#ffb27e'), Color('#fb6500')), # orange
    (Color('#beffcf'), Color('#33FF66')), # lime green
]

COLOR_HEX_LISTS = []
for color in COLORS:
    color_gradient = color[0].range_to(color[1], numBins)
    hex_list = [c.get_hex_l() for c in color_gradient]
    COLOR_HEX_LISTS.append(hex_list)

numColors = len(COLOR_HEX_LISTS)

MARKERS = [
    'D', 'o', 'v', 's', '^', '<',
    '>', '*', 'h', 'H', 'd', 'X'
]
numMarkers = len(MARKERS)
