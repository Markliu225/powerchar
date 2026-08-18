"""The paper's shared colour palette — ONE source for every figure in the repo.

A muted green/steel/orange set (the blues run navy -> steel -> cyan, the greens green -> mint, the
warms orange -> clay). Import it, never re-type a hex: figures that share a meaning must share a
colour, and the only way to guarantee that across a dozen scripts is a single table.

  * PAL          the named slots
  * CLASS_C      the seven Table-I workload classes, in canonical P:D order — so a class keeps its
                 colour in every figure that draws classes (validation curves, data-hall, profit)
  * MODEL_C      the four models under test, in size order
  * BAND_C       the P:D bands (decode-heavy / balanced / prefill-heavy)
  * OK / BAD / HL   semantic pair (capped vs TDP, i.e. good vs bad) + the highlight for crossovers
  * INK/INK2/MUTE/GRID   the neutral chrome, unchanged from the repo's earlier figures

The set is deliberately low-chroma, which costs separability: the pastels sit at 1.5-1.8:1 against
white and the orange/green pair is weak under deuteranopia (OKLab dE 5.3). Figures that draw more
than three series therefore carry a second channel — a distinct marker shape per series — and put
their numbers in _ink(), the same hue stepped down to a readable lightness (a flat clamp was tried
first and made the mint and green labels the same green, so the step is proportional).
"""
import colorsys

PAL = dict(navy="#4A5F7E", steel="#719AAC", cyan="#94C6CD",           # blues, dark -> light
           green="#72B063", mint="#B8DBB3",                           # greens, dark -> light
           orange="#E29135", sand="#EFC38A", clay="#B5615A",          # warms, dark -> light
           gray="#8C8C86")                                            # neutral band

# seven classes, canonical order = the taxonomy's P:D order (Reasoning 0.83 -> Code completion 110.7)
CLASS_C = [PAL["navy"], PAL["orange"], PAL["green"], PAL["cyan"], PAL["clay"], PAL["steel"],
           PAL["mint"]]
CLASS_ORDER = ["推理", "助手API", "多模态图文", "对话",
               "长上下文对话", "Agentic工具调用",
               "代码补全"]      # workload_classes.csv keys, same order as CLASS_C
CLASS_OF = dict(zip(CLASS_ORDER, CLASS_C))       # ALWAYS key by class, never by list position: the
                                                 # profit figure sorts by demand share, and a class
                                                 # that changed colour between figures would be a bug
MODEL_C = [PAL["steel"], PAL["orange"], PAL["green"], PAL["navy"]]    # 1.5B -> 7B
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
BAND_C = {"decode-heavy": PAL["clay"], "balanced": PAL["gray"], "prefill-heavy": PAL["steel"]}
PRE_C, DEC_C = PAL["steel"], PAL["orange"]      # phase pair, where both phases share a panel
OK, BAD, HL = PAL["green"], PAL["clay"], PAL["orange"]                # capped / TDP / highlight
INK, INK2, MUTE, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def lum(c):
    """HLS lightness of a '#rrggbb' string — the test for 'too pale to carry a white marker edge'."""
    return colorsys.rgb_to_hls(*(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)))[1]


def ink(c, k=0.58, cap=0.46):
    """A series colour stepped down for TEXT: same hue and saturation, lightness SCALED (not clamped)
    so the pale slots become readable on white while staying lighter than their darker siblings."""
    r, g, b = (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, sat = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, min(l * k, cap), sat)
