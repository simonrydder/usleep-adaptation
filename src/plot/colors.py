import seaborn as sns

pastel = sns.color_palette("pastel", 10)
bright = sns.color_palette("bright", 10)

BASE_COLOR = {
    "dod-h": pastel[0],
    "dod-o": pastel[1],
    "eesm19": pastel[2],
    "eesm23": pastel[3],
    "isruc_sg1": pastel[4],
    "isruc_sg2": pastel[5],
    "isruc_sg3": pastel[6],
    "mass_c1": pastel[7],
    "mass_c2": pastel[8],
    "svuh": pastel[9],
}

HIGHLIGHT_COLOR = {
    "dod-h": bright[0],
    "dod-o": bright[1],
    "eesm19": bright[2],
    "eesm23": bright[3],
    "isruc_sg1": bright[4],
    "isruc_sg2": bright[5],
    "isruc_sg3": bright[6],
    "mass_c1": bright[7],
    "mass_c2": bright[8],
    "svuh": bright[9],
}
