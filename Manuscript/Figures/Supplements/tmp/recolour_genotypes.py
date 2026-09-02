"""
Apply the genotype colour scheme to a figure SVG, in place.

    +/+  lightseagreen  #20B2AA
    +/-  orchid         #DA70D6
    -/-  orange         #FFA500

The wild-type series in these seaborn exports declares no `fill:` at all and relies on
the SVG default of black, so it is identified by "has a partial fill-opacity but no fill"
rather than by colour. Two things must be protected:

  * the MAP-Map panels, which use pure green (#00ff00) / magenta (#ff00ff) for
    increased / decreased activity;
  * the MAP-Map legend text, which reads "... atp1a3a-/- vs atp1a3a+/+" and therefore
    matches a naive search for genotype labels.

Usage:  python recolour_genotypes.py Figure_DarkFlash.svg [--dry-run]
"""
import sys, re, shutil
import xml.etree.ElementTree as ET

WT, HET, MUT = "#20B2AA", "#DA70D6", "#FFA500"
HET_OLD = ["#d70000", "#bc1b1b"]
MUT_OLD = ["#8c3cff", "#9054e7", "#8d3aff", "#800080"]
MAP_COLOURS = ["#00ff00", "#ff00ff"]          # never touch these
PROTECT_TEXT = ["activity"]                   # substrings marking non-genotype labels

SVG = "http://www.w3.org/2000/svg"
for pre, uri in [("", SVG), ("xlink", "http://www.w3.org/1999/xlink"),
                 ("inkscape", "http://www.inkscape.org/namespaces/inkscape"),
                 ("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd")]:
    ET.register_namespace(pre, uri)

tag = lambda e: e.tag.split("}")[-1]
sty = lambda e: (e.get("style") or "")


def paint(el, colour):
    """Force fill (and any stroke that is already coloured) to `colour`."""
    s = sty(el)
    if "fill:" in s:
        s = re.sub(r"fill:\s*#[0-9a-fA-F]{6}", "fill:" + colour, s)
    else:
        s = "fill:%s;%s" % (colour, s)
    el.set("style", s)
    if el.get("fill", "").startswith("#"):
        el.set("fill", colour)


def main(path, dry=False):
    if not dry:
        shutil.copy(path, path + ".bak")
    tree = ET.parse(path)
    root = tree.getroot()

    parent = {}
    for p in root.iter():
        for c in p:
            parent[c] = p

    def ancestors(e):
        out = []
        while e in parent:
            e = parent[e]
            out.append(e)
        return out

    # subtrees containing MAP-Map colours are off limits
    def has_map(g):
        for e in g.iter():
            s = sty(e).lower() + " " + (e.get("fill") or "").lower()
            if any(c in s for c in MAP_COLOURS):
                return True
        return False
    def has_geno(g):
        for e in g.iter():
            s = sty(e).lower()
            if any(c in s for c in HET_OLD + MUT_OLD + [WT.lower(), HET.lower(), MUT.lower()]):
                return True
        return False
    # Protect only groups that contain MAP-Map colours *and no genotype plot colours* —
    # that isolates the MAP-Map panel without catching the root group, which contains both.
    protected = set()
    for g in root.iter():
        if tag(g) == "g" and has_map(g) and not has_geno(g):
            for e in g.iter():
                protected.add(id(e))

    n_wt = n_col = n_lbl = 0

    # 1. het / mut, identified unambiguously by colour
    for el in root.iter():
        s = sty(el)
        if not s:
            continue
        new = s
        for c in HET_OLD:
            new = re.sub(c, HET, new, flags=re.I)
        for c in MUT_OLD:
            new = re.sub(c, MUT, new, flags=re.I)
        if new != s:
            el.set("style", new)
            n_col += 1

    # 2. wild-type plot elements: partial alpha, no fill declared, outside the MAP-Map
    for el in root.iter():
        s = sty(el)
        if not s or "fill:" in s or id(el) in protected:
            continue
        if re.search(r"fill-opacity:\s*0\.\d", s):
            el.set("style", "fill:%s;%s" % (WT, s))
            n_wt += 1

    # 3. genotype text labels, skipping the MAP-Map legend
    norm = lambda s: s.replace("\n", "").replace(" ", "")
    for el in root.iter():
        if tag(el) != "text":
            continue
        raw = "".join(el.itertext())
        if any(p in raw.lower() for p in PROTECT_TEXT):
            continue
        s = norm(raw)
        colour = (WT if "atp1a3a+/+" in s else
                  HET if "atp1a3a+/-" in s else
                  MUT if "atp1a3a-/-" in s else None)
        if colour:
            for nd in el.iter():
                paint(nd, colour)
            n_lbl += 1

    if not dry:
        tree.write(path, encoding="utf-8", xml_declaration=True)
    print("%-34s het/mut styles:%5d  wild-type:%5d  labels:%2d%s"
          % (path, n_col, n_wt, n_lbl, "  (dry run)" if dry else ""))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], dry="--dry-run" in sys.argv)
