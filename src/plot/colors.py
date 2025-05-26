import seaborn as sns

pastel = sns.color_palette("pastel", 10)
bright = sns.color_palette("bright", 10)

BASE_COLOR = {
    "dod_h": pastel[0],
    "dod_o": pastel[1],
    "eesm19": pastel[2],
    "isruc_sg1": pastel[3],
    "isruc_sg2": pastel[4],
    "isruc_sg3": pastel[5],
    "mass_c1": pastel[6],
    "mass_c3": pastel[8],
    "svuh": pastel[9],
}

HIGHLIGHT_COLOR = {
    "dod_h": bright[0],
    "dod_o": bright[1],
    "eesm19": bright[2],
    "isruc_sg1": bright[3],
    "isruc_sg2": bright[4],
    "isruc_sg3": bright[5],
    "mass_c1": bright[6],
    "mass_c3": bright[8],
    "svuh": bright[9],
}
