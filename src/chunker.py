"""One chunker for training, evaluation AND the Streamlit app (no train/serve skew)."""
import re
WIN, STRIDE = 100, 50          # words; LD spans are median 49 words, so most fit whole in one window

def chunk_text(text, win=WIN, stride=STRIDE):
    """Yield (char_start, char_end, chunk_text) over overlapping word windows."""
    toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out = []
    for i in range(0, max(len(toks) - win, 0) + 1, stride):
        seg = toks[i:i + win]
        if not seg: break
        a, b = seg[0][0], seg[-1][1]
        out.append((a, b, re.sub(r"\s+", " ", text[a:b])))
    if toks and out and out[-1][1] < toks[-1][1]:          # tail
        a = toks[max(len(toks) - win, 0)][0]; b = toks[-1][1]
        out.append((a, b, re.sub(r"\s+", " ", text[a:b])))
    return out
