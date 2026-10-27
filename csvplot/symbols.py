"""Math symbols, super/subscripts and safe rendering of matplotlib mathtext.

Text between ``$...$`` is drawn as math (matplotlib's LaTeX subset), e.g.
``$\\Delta R/R$`` or ``cm$^{-1}$``.
"""

from __future__ import annotations

from matplotlib.mathtext import MathTextParser
from PySide6.QtWidgets import QApplication, QLineEdit, QMenu, QToolButton

# (menu label, template). "{}" is replaced by the selected text, or by the
# default after "|" when nothing is selected.
FORMATTING = [
    ("x² Superscript", "^{{{}}}|2"),
    ("xᵢ Subscript", "_{{{}}}|i"),
    ("a⁄b Fraction", r"\frac{{{}}}{{b}}|a"),
    ("√x Square root", r"\sqrt{{{}}}|x"),
    ("x̄ Bar", r"\bar{{{}}}|x"),
    ("x̂ Hat", r"\hat{{{}}}|x"),
    ("ẋ Dot", r"\dot{{{}}}|x"),
    ("𝐱 Bold", r"\mathbf{{{}}}|x"),
    ("x Upright (roman)", r"\mathrm{{{}}}|x"),
]

GREEK_LOWER = [
    ("α", r"\alpha"), ("β", r"\beta"), ("γ", r"\gamma"), ("δ", r"\delta"),
    ("ε", r"\epsilon"), ("ζ", r"\zeta"), ("η", r"\eta"), ("θ", r"\theta"),
    ("κ", r"\kappa"), ("λ", r"\lambda"), ("μ", r"\mu"), ("ν", r"\nu"),
    ("ξ", r"\xi"), ("π", r"\pi"), ("ρ", r"\rho"), ("σ", r"\sigma"),
    ("τ", r"\tau"), ("φ", r"\phi"), ("χ", r"\chi"), ("ψ", r"\psi"), ("ω", r"\omega"),
]
GREEK_UPPER = [
    ("Γ", r"\Gamma"), ("Δ", r"\Delta"), ("Θ", r"\Theta"), ("Λ", r"\Lambda"),
    ("Ξ", r"\Xi"), ("Π", r"\Pi"), ("Σ", r"\Sigma"), ("Φ", r"\Phi"),
    ("Ψ", r"\Psi"), ("Ω", r"\Omega"),
]
OPERATORS = [
    ("±", r"\pm"), ("∓", r"\mp"), ("×", r"\times"), ("·", r"\cdot"), ("÷", r"\div"),
    ("≈", r"\approx"), ("≠", r"\neq"), ("≤", r"\leq"), ("≥", r"\geq"), ("∝", r"\propto"),
    ("∞", r"\infty"), ("∂", r"\partial"), ("∇", r"\nabla"), ("∫", r"\int"), ("∑", r"\sum"),
    ("→", r"\rightarrow"), ("←", r"\leftarrow"), ("↔", r"\leftrightarrow"),
    ("⟨ ⟩", r"\langle x \rangle"),
]
UNITS = [
    ("° degree", r"^\circ"), ("Å ångström", r"\AA"), ("ħ h-bar", r"\hbar"),
    ("ℓ ell", r"\ell"), ("µm", r"\mu\mathrm{m}"), ("cm⁻¹", r"\mathrm{cm}^{-1}"),
]

_parser = MathTextParser("path")


def is_valid_mathtext(text: str) -> bool:
    if text.count("$") - text.count(r"\$") < 2:
        return True  # no math segment; drawn as plain text
    try:
        _parser.parse(text)
    except ValueError:
        return False
    return True


def safe_text(text: str) -> tuple[str, bool]:
    """Return (text to draw, ok). Invalid math is shown literally instead of failing."""
    if is_valid_mathtext(text):
        return text, True
    return text.replace(r"\$", "$").replace("$", r"\$"), False


def _inside_math(text: str, cursor: int) -> bool:
    """True if ``cursor`` sits between an opening and closing ``$``."""
    before = text[:cursor].replace(r"\$", "")
    return before.count("$") % 2 == 1


def insert_math(edit: QLineEdit, template: str) -> None:
    """Insert ``template`` at the cursor (wrapping any selection) as math."""
    body, _, default = template.partition("|")
    selected = edit.selectedText()
    # Templates with a "|default" part have a slot for the selection.
    content = body.format(selected or default) if default else body
    text = edit.text()
    start = edit.selectionStart() if edit.hasSelectedText() else edit.cursorPosition()
    end = start + len(selected)
    piece = content if _inside_math(text, start) else f"${content}$"
    new = text[:start] + piece + text[end:]
    edit.setText(new)
    edit.setCursorPosition(start + len(piece))
    edit.setFocus()
    edit.editingFinished.emit()


class SymbolButton(QToolButton):
    """'Symbols' menu that inserts math into the label field used most recently."""

    def __init__(self, edits: list[QLineEdit], parent=None):
        super().__init__(parent)
        self.setText("Insert symbol / super, subscript")
        self.setToolTip("Inserts into the last label field you clicked.\n"
                        "Text between $…$ is drawn as math, e.g. $\\Delta R/R$ or cm$^{-1}$.\n"
                        "Select text first to make it a superscript, subscript, etc.")
        self.setPopupMode(QToolButton.InstantPopup)
        self.setObjectName("symbols")
        self.edits = edits
        self.target: QLineEdit | None = edits[0] if edits else None
        QApplication.instance().focusChanged.connect(self._on_focus)

        menu = QMenu(self)
        fmt = menu.addMenu("Superscript, subscript, …")
        for label, template in FORMATTING:
            fmt.addAction(label, lambda t=template: self._insert(t))
        for title, items in (("Greek (lower case)", GREEK_LOWER), ("Greek (upper case)", GREEK_UPPER),
                             ("Operators && arrows", OPERATORS), ("Units && constants", UNITS)):
            sub = menu.addMenu(title)
            for glyph, latex in items:
                sub.addAction(f"{glyph}    {latex}", lambda t=latex: self._insert(t))
        self.setMenu(menu)

    def _on_focus(self, _old, new) -> None:
        if new in self.edits:
            self.target = new

    def _insert(self, template: str) -> None:
        if self.target is not None:
            insert_math(self.target, template)


# ------------------------------------------------------------- full LaTeX mode
_LATEX_SPECIALS = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
                   "^": r"\^{}", "~": r"\textasciitilde{}"}


def latex_available() -> tuple[bool, str]:
    """Whether matplotlib's usetex can run here, and what is missing if not."""
    import shutil

    missing = [tool for tool in ("latex", "dvipng") if shutil.which(tool) is None]
    if missing:
        return False, ("Full LaTeX needs a LaTeX installation (e.g. TeX Live or MiKTeX) "
                       f"with {' and '.join(missing)} on the PATH.")
    return True, ""


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters outside $...$ so plain names like
    'dR_over_R' work; math segments are passed through unchanged."""
    parts, out = text.split("$"), []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # outside math
            part = "".join(_LATEX_SPECIALS.get(ch, ch) for ch in part)
        out.append(part)
    if len(parts) % 2 == 0:  # unmatched $: treat the last one literally
        return "$".join(out[:-1]) + r"\$" + out[-1]
    return "$".join(out)
