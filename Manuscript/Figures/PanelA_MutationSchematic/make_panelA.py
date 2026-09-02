"""
Panel A of Figure 1: the atp1a3a allele, in genomic and protein frames of reference.

Draws three tiers:
  1. gene model  - 21 exons, 8 bp deletion in exon 8, premature stop in exon 9
  2. sequence    - the target site, with the deleted bases and the resulting frameshift
  3. protein     - membrane topology of wild-type (10 TM helices) beside the truncated
                   mutant, which retains only M1-M2 before 78 frameshifted residues

Transmembrane helix boundaries are taken from human ATP1A3 (UniProt P13637, 1013 aa)
and drawn on the 1023 aa zebrafish Atp1a3a axis; the two proteins are ~92% identical so
the positions are approximate to within a few residues.

Run:  python make_panelA.py   ->  Figure_PanelA_MutationSchematic.svg
"""

OUT = "Figure_PanelA_MutationSchematic.svg"

# ----------------------------------------------------------------- data
N_EXONS = 21
MUT_EXON = 8            # 8 bp deletion
STOP_EXON = 9           # premature termination codon

# Coding length of each exon (nt), from the canonical transcript ENSDART00000104950
# (Ensembl GRCz11). Exon 1 is almost entirely 5\'UTR (6 nt coding) and exon 21 almost
# entirely 3\'UTR (29 nt coding); the CDS totals 3072 nt = 1023 aa + stop, matching the
# protein length quoted in the text. Exon widths are drawn proportional to these values
# so that position along the gene track equals position along the protein.
EXON_CDS = [6, 117, 60, 204, 114, 135, 118, 269, 199, 245, 193,
            313, 151, 169, 155, 124, 146, 131, 102, 92, 29]
CDS_TOTAL = sum(EXON_CDS)                 # 3072

# Read off the original panel A: the underlined gRNA protospacer is the full 20 nt
# ACTGGAGGCCGGACCTCTGG, and the 8 deleted bases (ACTGGAGG) are its 5' half.
SEQ_WT  = [("GGGG", "flank"), ("ACTGGAGG", "deleted"), ("CCGGACCTCTGG", "kept"), ("CCTTTC", "flank")]
SEQ_MUT = [("GGGG", "flank"), ("--------", "gap"),     ("CCGGACCTCTGG", "kept"), ("CCTTTC", "flank")]
PROTOSPACER_BLOCKS = (1, 2)   # blocks spanned by the 20 nt gRNA target

AA_WT = 1023            # zebrafish Atp1a3a
AA_IDENT = 272          # residues identical to wild-type
AA_NOVEL = 78           # frameshifted residues
AA_MUT = AA_IDENT + AA_NOVEL
AA_IDENTITY_HS = 92     # % amino acid conservation with human ATP1A3 (as quoted in the text)

TM = [(91, 107), (119, 139), (282, 303), (307, 345), (767, 786),
      (794, 809), (840, 862), (905, 927), (943, 966), (975, 995)]
D_PHOS = 366            # aspartylphosphate intermediate

# ----------------------------------------------------------------- style
# Genotype palette (matplotlib named CSS colours), shared across the figures:
#   +/+  lightseagreen  #20B2AA      +/-  orchid  #DA70D6      -/-  orange  #FFA500
C_WT     = "#20B2AA"    # lightseagreen - wild-type allele / protein
C_HET    = "#DA70D6"    # orchid        - heterozygote (not depicted in this panel)
C_MUT    = "#FFA500"    # orange        - mutant allele / everything downstream of the lesion

C_EXON   = C_WT         # exons / wild-type protein
C_EXON_L = "#8ad9d4"    # light tint of lightseagreen
C_HIT    = C_MUT        # the lesion, and everything downstream of it
C_HIT_L  = "#ffd9a0"    # light tint of orange
# Pure #FFA500 is thin against white at small type, so annotation *text* uses a darker
# shade of the same hue while all graphical elements keep Tod's exact colour.
C_HIT_TX = "#B36B00"
C_GENE   = "#9aa3a8"    # neutral grey for the gene model - teal/orange are reserved
                        # for the protein cartoons and for the lesion itself
C_GRNA   = "#4C72B0"    # gRNA protospacer - deliberately outside the genotype palette
C_STOP   = "#CC0000"    # premature stop sign
C_MEM    = "#e9e4d8"    # lipid bilayer band
C_INK    = "#1a1a1a"
C_GREY   = "#8a8a8a"
FONT     = "Arial, Helvetica, sans-serif"

W, H = 190.0, 112.0

parts = []
def add(s): parts.append(s)

def text(x, y, s, size=3.2, weight="normal", fill=C_INK, anchor="start",
         family=FONT, style=""):
    add(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'style="{style}">{s}</text>')

def rect(x, y, w, h, fill, rx=0.4, stroke="none", sw=0.3, opacity=1.0):
    add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

def line(x1, y1, x2, y2, stroke=C_INK, sw=0.35, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

def path(d, stroke=C_INK, sw=0.4, fill="none", cap="round", dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="{cap}" stroke-linejoin="round"{da}/>')

# =====================================================================
# TIER 1 - gene model
# =====================================================================
GX0, GX1, GY = 10.0, 180.0, 20.0
EH = 4.6

text(2, 9.2, "atp1a3a", size=4.4, weight="bold", style="font-style:italic")
add(f'<text x="20.5" y="9.2" font-family="{FONT}" font-size="3.0" fill="{C_GREY}">'
    f'{N_EXONS} exons  ·  {AA_IDENTITY_HS}% amino acid conservation with human '
    f'<tspan font-style="italic">ATP1A3</tspan></text>')

line(GX0, GY + EH / 2, GX1, GY + EH / 2, stroke=C_GENE, sw=0.5)
# small chevrons along the introns to give the line some direction
for i in range(12):
    cx = GX0 + (GX1 - GX0) * (i + 0.5) / 12
    path(f"M{cx-0.7:.2f},{GY+EH/2-0.9:.2f} L{cx+0.5:.2f},{GY+EH/2:.2f} "
         f"L{cx-0.7:.2f},{GY+EH/2+0.9:.2f}", stroke="#ffffff", sw=0.3)

# exon widths proportional to coding length, so that x-position along the track is
# proportional to position along the 1023 aa protein
GAP = 0.9
scale = (GX1 - GX0 - GAP * (N_EXONS - 1)) / CDS_TOTAL
widths = [max(c * scale, 0.55) for c in EXON_CDS]
gap = GAP
x = GX0
exon_x = []
for i, w in enumerate(widths, start=1):
    exon_x.append((x, w))
    fill, stroke = C_GENE, "none"
    rect(x, GY, w, EH, fill, rx=0.5, stroke=stroke, sw=0.35)
    x += w + gap

for i, lbl in ((MUT_EXON, "8"), (STOP_EXON, "9")):
    ex, ew = exon_x[i - 1]
    text(ex + ew / 2, GY + EH + 3.0, lbl, size=2.7, fill=C_GREY, anchor="middle")

def cds_to_x(nt):
    """x coordinate of a coding position (nt) on the exon track."""
    run = 0
    for i, c in enumerate(EXON_CDS):
        if run + c >= nt:
            ex, ew = exon_x[i]
            return ex + ew * (nt - run) / c
        run += c
    return exon_x[-1][0] + exon_x[-1][1]

# lesion marker: the deletion truncates the wild-type sequence after codon 272
mx = cds_to_x(AA_IDENT * 3)
path(f"M{mx:.2f},{GY-1.0:.2f} L{mx-1.5:.2f},{GY-3.6:.2f} L{mx+1.5:.2f},{GY-3.6:.2f} Z",
     fill=C_HIT, stroke="none")
line(mx, GY - 3.6, mx, GY - 6.5, stroke=C_HIT, sw=0.45)
text(mx - 2.0, GY - 7.4, "8 bp deletion", size=3.1, weight="bold", fill=C_HIT_TX, anchor="end")

# stop marker over exon 9
sx = cds_to_x(AA_MUT * 3)
line(sx, GY - 1.0, sx, GY - 5.2, stroke=C_STOP, sw=0.4, dash="0.9,0.7")
import math as _m
_oct = " ".join(f"{sx+1.7*_m.cos(_m.radians(a+22.5)):.2f},{GY-6.4+1.7*_m.sin(_m.radians(a+22.5)):.2f}"
                for a in range(0, 360, 45))
add(f'<polygon points="{_oct}" fill="{C_STOP}" stroke="#ffffff" stroke-width="0.35"/>')
text(sx + 2.9, GY - 7.4, "premature stop codon", size=3.1, weight="bold", fill=C_STOP)

# =====================================================================
# TIER 2 - target site sequence
# =====================================================================
SY = 38.0
bx0, bx1 = mx - 40.0, mx + 40.0
# The sequence below is a 30 nt window, not the whole 269 nt exon: the 8 deleted bases
# begin at CDS nt AA_IDENT*3 + 1, and the window starts 4 nt upstream of them.
WIN_LEN = sum(len(s) for s, _ in SEQ_WT)          # 30 nt
WIN_CDS0 = AA_IDENT * 3 - 3                       # 4 nt of flank before the deletion
WIN_CDS1 = WIN_CDS0 + WIN_LEN
wx0, wx1 = cds_to_x(WIN_CDS0), cds_to_x(WIN_CDS1)

# mark the expanded window on exon 8
rect(wx0, GY - 0.5, max(wx1 - wx0, 0.35), EH + 1.0, C_MUT, rx=0.15,
     stroke=C_INK, sw=0.35)
path(f"M{wx0:.2f},{GY+EH:.2f} L{bx0+2:.2f},{SY-4.5:.2f}", stroke=C_GREY, sw=0.3, dash="1,0.8")
path(f"M{wx1:.2f},{GY+EH:.2f} L{bx1-2:.2f},{SY-4.5:.2f}", stroke=C_GREY, sw=0.3, dash="1,0.8")
rect(bx0, SY - 4.5, bx1 - bx0, 21.0, "#fbfbf9", rx=1.0, stroke="#dcdcdc", sw=0.35)

CHW = 2.04   # Courier advance (0.6 em at 2.9) + 0.30 letter-spacing
def draw_seq(y, blocks, label, label_fill=C_INK, protospacer=True):
    text(bx0 + 3.0, y, label, size=2.9, weight="bold", fill=label_fill, anchor="end")
    xx = bx0 + 4.5
    spans = []
    for s, kind in blocks:
        w = len(s) * CHW
        if kind == "deleted":
            rect(xx - 0.3, y - 3.0, w + 0.2, 4.0, C_HIT_L, rx=0.4, opacity=0.75)
        spans.append((xx, w, kind))
        fill = C_HIT if kind in ("deleted", "gap") else C_INK
        add(f'<text x="{xx:.2f}" y="{y:.2f}" font-family="Courier New, monospace" '
            f'font-size="2.9" letter-spacing="0.30" fill="{fill}" '
            f'font-weight="{"bold" if kind in ("deleted","gap") else "normal"}">{s}</text>')
        if kind == "deleted":
            line(xx - 0.2, y - 1.0, xx + w - 0.4, y - 1.0, stroke=C_HIT, sw=0.4)
        xx += w
    if protospacer:
        a = spans[PROTOSPACER_BLOCKS[0]][0] - 0.3
        b = spans[PROTOSPACER_BLOCKS[1]][0] + spans[PROTOSPACER_BLOCKS[1]][1] - 0.5
        line(a, y + 1.15, b, y + 1.15, stroke=C_GRNA, sw=0.55)
        line(a, y + 0.65, a, y + 1.65, stroke=C_GRNA, sw=0.55)
        line(b, y + 0.65, b, y + 1.65, stroke=C_GRNA, sw=0.55)
    return xx

text(bx0 + 4.5, SY - 1.2, f"CRISPR target site: {WIN_LEN} nt of exon 8 (269 nt)",
     size=2.8, weight="bold", fill=C_GREY)
end_wt = draw_seq(SY + 4.6, SEQ_WT, "WT", C_INK)
end_mut = draw_seq(SY + 10.2, SEQ_MUT, "mut", C_HIT_TX, protospacer=False)
line(bx0 + 4.5, SY + 14.0, bx0 + 9.5, SY + 14.0, stroke=C_GRNA, sw=0.55)
text(bx0 + 10.5, SY + 14.9, "gRNA target (20 nt)", size=2.5, fill=C_GREY)
rect(bx0 + 34.0, SY + 13.0, 4.0, 1.7, C_HIT_L, rx=0.3, opacity=0.75)
text(bx0 + 39.5, SY + 14.9, "8 bp deleted", size=2.5, fill=C_GREY)

text(bx1 + 2.0, SY + 7.4, "→ frameshift", size=3.2, weight="bold", fill=C_HIT_TX)


# =====================================================================
# TIER 3 - protein topology
# =====================================================================
MEM_TOP, MEM_BOT = 72.0, 84.0
def topology(x0, x1, helices, title, subtitle, novel_tail=False, step_ref=None):
    """Draw a membrane band with helices crossing it, N-term at left."""
    rect(x0, MEM_TOP, x1 - x0, MEM_BOT - MEM_TOP, C_MEM, rx=0.0)
    line(x0, MEM_TOP, x1, MEM_TOP, stroke="#cfc7b4", sw=0.3)
    line(x0, MEM_BOT, x1, MEM_BOT, stroke="#cfc7b4", sw=0.3)

    n = len(helices)
    span = (x1 - x0) - 8.0
    step = step_ref if step_ref else span / max(n, 1)
    hw = min(3.2, step * 0.55)
    xs = [x0 + 4.0 + step * (i + 0.5) for i in range(n)]
    for i, hx in enumerate(xs):
        rect(hx - hw / 2, MEM_TOP - 1.2, hw, (MEM_BOT - MEM_TOP) + 2.4,
             C_EXON, rx=1.4, stroke="#ffffff", sw=0.35)
        text(hx, MEM_BOT + 3.0, f"M{helices[i]}", size=2.3, fill=C_GREY, anchor="middle")
    # connecting loops, alternating above / below the membrane
    for i in range(n - 1):
        a, b = xs[i], xs[i + 1]
        mid = (a + b) / 2
        big = (helices[i], helices[i + 1]) == (4, 5)
        if i % 2 == 0:   # extracellular
            top = MEM_TOP - (3.0 if not big else 3.0)
            path(f"M{a:.2f},{MEM_TOP-1.2:.2f} C{a:.2f},{top:.2f} {b:.2f},{top:.2f} {b:.2f},{MEM_TOP-1.2:.2f}",
                 stroke=C_EXON, sw=0.7)
        else:            # cytoplasmic
            depth = MEM_BOT + (4.0 if not big else 14.0)
            path(f"M{a:.2f},{MEM_BOT+1.2:.2f} C{a:.2f},{depth:.2f} {b:.2f},{depth:.2f} {b:.2f},{MEM_BOT+1.2:.2f}",
                 stroke=C_EXON, sw=0.7)
            if big:
                rect(mid - 13.0, MEM_BOT + 8.0, 26.0, 7.0, "#ffffff", rx=1.2,
                     stroke=C_EXON, sw=0.4)
                text(mid, MEM_BOT + 11.3, "N / P domains", size=2.5, anchor="middle", fill=C_EXON)
                text(mid, MEM_BOT + 14.0, f"catalytic Asp{D_PHOS}", size=2.2,
                     anchor="middle", fill=C_GREY)
    # N-terminus
    path(f"M{x0+1.0:.2f},{MEM_BOT+5.0:.2f} L{xs[0]:.2f},{MEM_BOT+1.2:.2f}", stroke=C_EXON, sw=0.7)
    text(x0 + 0.4, MEM_BOT + 6.6, "N", size=2.6, weight="bold", fill=C_EXON)
    if novel_tail:
        # 78 frameshifted residues as a red squiggle ending in a stop
        sx0 = xs[-1]
        d = f"M{sx0:.2f},{MEM_BOT+1.2:.2f}"
        for k in range(6):
            d += (f" q {2.2:.2f},{2.6 if k % 2 == 0 else -2.6:.2f} "
                  f"{4.4:.2f},0")
        path(d, stroke=C_HIT, sw=0.8)
        ex = sx0 + 26.4
        _o2 = " ".join(f"{ex+1.6*_m.cos(_m.radians(a+22.5)):.2f},"
                       f"{MEM_BOT+1.2+1.6*_m.sin(_m.radians(a+22.5)):.2f}"
                       for a in range(0, 360, 45))
        add(f'<polygon points="{_o2}" fill="{C_STOP}" stroke="#ffffff" stroke-width="0.35"/>')
        text(ex + 2.6, MEM_BOT + 2.2, "stop", size=2.5, weight="bold", fill=C_STOP)
        text(min(sx0 + 13.0, 176.0), MEM_BOT + 8.4, f"{AA_NOVEL} frameshifted residues",
             size=2.7, weight="bold", fill=C_HIT_TX, anchor="middle")
    else:
        path(f"M{xs[-1]:.2f},{MEM_BOT+1.2:.2f} L{x1-1.0:.2f},{MEM_BOT+5.0:.2f}", stroke=C_EXON, sw=0.7)
        text(x1 - 0.2, MEM_BOT + 6.6, "C", size=2.6, weight="bold", fill=C_EXON, anchor="end")
    text(x0, 65.0, title, size=3.4, weight="bold", style="font-style:italic")
    text(x0, 68.6, subtitle, size=2.7, fill=C_GREY)

text(11.5, 71.0, "out", size=2.4, fill=C_GREY, anchor="end")
text(11.5, 86.5, "in", size=2.4, fill=C_GREY, anchor="end")

topology(14, 108, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         "wild-type Atp1a3a", f"{AA_WT} aa · 10 transmembrane helices")
WT_STEP = ((108 - 14) - 8.0) / 10.0     # keep wild-type helix spacing in the mutant
topology(124, 182, [1, 2],
         "atp1a3a mutant", f"{AA_MUT} aa · {AA_IDENT} aa wild-type + {AA_NOVEL} novel",
         novel_tail=True, step_ref=WT_STEP)

# bracket showing how much of the protein is lost
line(112, 78, 120, 78, stroke=C_GREY, sw=0.4, dash="1,0.8")
path(f"M118,76.6 L120.4,78 L118,79.4", stroke=C_GREY, sw=0.4)

text(95, 108.0, "truncation removes M3–M10 and the entire catalytic core",
     size=2.8, fill=C_HIT_TX, anchor="middle", weight="bold")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" '
       f'viewBox="0 0 {W} {H}">\n<rect width="{W}" height="{H}" fill="#ffffff"/>\n'
       + "\n".join(parts) + "\n</svg>\n")
open(OUT, "w", encoding="utf-8").write(svg)
print(f"wrote {OUT}")
