ESS Thesis Atlas - local copy

Open in VS Code:
1. File > Open Folder... and choose this folder (ESS_Thesis_Atlas).
2. Install the extension "Live Server" (by Ritwick Dey) if you don't have it.
3. Right-click index.html > "Open with Live Server". It opens in your browser.
   To keep it inside VS Code: Ctrl+Shift+P > "Simple Browser: Show" and paste
   the address Live Server shows (e.g. http://127.0.0.1:5500/index.html).

Without VS Code: open a terminal in this folder and run
   python -m http.server 8000
then open http://localhost:8000 in a browser.

Double-clicking index.html also works for the network and the notes, but the
PDF viewer needs one of the two servers above (browsers block it on file://).
An internet connection is needed: the page loads d3, pdf.js, marked and MathJax
from public CDNs.
