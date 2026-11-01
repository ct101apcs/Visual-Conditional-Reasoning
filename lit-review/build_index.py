"""Build lit-review/papers.md (one table of every cited work) and papers/MANIFEST.tsv (local PDF names).

  python3 lit-review/build_index.py            # rewrite papers.md and papers/MANIFEST.tsv
  python3 lit-review/build_index.py --fetch    # download missing PDFs listed in MANIFEST.tsv from arxiv.org

Sources: the "## Bibliography" section of each note in research_notes/subject-conditioned-object-reasoning/.
The five notes use five citation styles; parse() extracts arXiv ID, first author, year and title from each.
PDFs are not tracked in git (1.3 GB); MANIFEST.tsv keeps their file names so --fetch restores the folder.
"""
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(HERE, "research_notes", "subject-conditioned-object-reasoning")
PAPERS = os.path.join(HERE, "papers")
THEMES = {
    "01": "Referring, grounding and visual prompting",
    "02": "Spatial, relational and region benchmarks",
    "03": "Rule adherence, priors and logic",
    "04": "Distractors, editing and the data source",
    "05": "Benchmark methodology and statistics",
}
# arXiv IDs of the paper's reference list (paper_scaffold.md, "References (<= 20)"), in that order;
# one entry may hold several IDs; None = no arXiv ID (ref 17, the temporal-attention PDF)
PAPER_REFS = ["2606.30393", "2502.02589", "1405.0312", "2502.15563", "2406.16860", "2406.01584",
              ("2306.00968", "2601.05244"), "2511.16719", "2310.11441", "2001.06927", "2310.01846", "2405.14156",
              "2604.12119", "2501.09425", "2511.06146", "2607.23386", None, "1902.09506", "2411.00640", "2511.21631"]
# works the paper cites that no note covers (checked on arXiv 2026-11-01; see paper_scaffold.md, refs 10-12)
EXTRA = [dict(id="2310.01846", author="Li", year="2023", theme="03",
              title="Benchmarking and Improving Generator-Validator Consistency of Language Models (ICLR 2024)"),
         dict(id="2405.14156", author="Zhang", year="2024", theme="03",
              title="Unveiling the Tapestry of Consistency in Large Vision-Language Models (ConBench, NeurIPS 2024)"),
         dict(id="2606.30393", author="(ours)", year="2026", theme="04",
              title="SADL: Subject-Aware Distractor Localization (the earlier attempt this project rebuilds)")]
ID = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?![\d])")
YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def parse(line):
    """One bibliography line -> dict(id, author, year, title) or None."""
    s = re.sub(r"^\s*(?:-|\d+\.)\s+", "", line.strip())
    s_noid = ID.sub("", s)
    m = ID.search(s)
    aid = m.group(1) if m else ""
    if " | " in s:                                    # 02: ID | Author | Year | Title
        parts = [p.strip() for p in s.split(" | ")]
        return dict(id=aid, author=parts[1], year=parts[2], title=" | ".join(parts[3:]))
    q = None if ID.match(s) else (re.search(r"[\"“](.+?)[\"”]", s) or re.search(r"\*(.+?)\*", s))  # 04, 05
    if q:
        title = q.group(1).rstrip(".")
        author = re.split(r"[,.]", s, 1)[0].strip()
    else:                                             # 01 "Author et al. YEAR, Title, ID"; 03 "ID Author YEAR, Title"
        mm = re.match(r"(.+?)\s+((?:19|20)\d\d)[a-z]?,\s*(.+)$", s_noid.strip())
        if not mm:
            return None
        author, title = mm.group(1), mm.group(3)
        title = re.sub(r"\s*\[[^\]]*\]\s*$", "", title).strip().rstrip(",").strip()
    author = re.sub(r"\s+et al\.?", "", author).split(",")[0].strip()
    y = YEAR.search(s_noid)
    title = re.sub(r"\s*\[(DL|DL-other|-)\]", "", title).strip()
    return dict(id=aid, author=author, year=y.group(1) if y else "", title=title)


def load():
    works, order = {}, []
    for fn in sorted(os.listdir(NOTES)):
        if not fn.endswith(".md"):
            continue
        theme = fn[:2]
        text = open(os.path.join(NOTES, fn)).read()
        bib = text.split("## Bibliography", 1)[-1]
        for line in bib.splitlines():
            if not re.match(r"\s*(?:-|\d+\.)\s+\S", line):
                continue
            w = parse(line)
            if not w or not (w["id"] or w["title"]):
                continue
            key = w["id"] or w["title"].lower()
            if key not in works:
                works[key] = dict(w, themes=[theme])
                order.append(key)
            else:
                for f in ("author", "year", "title"):          # fill fields a terser citation left empty
                    works[key][f] = works[key][f] or w[f]
                if theme not in works[key]["themes"]:
                    works[key]["themes"].append(theme)
    for w in EXTRA:
        if w["id"] not in works:
            works[w["id"]] = dict(w, themes=[w["theme"]])
            order.append(w["id"])
    return works, order


def local_pdfs():
    """arXiv ID -> file name for PDFs in papers/; also file names without an ID."""
    by_id, other = {}, []
    if os.path.isdir(PAPERS):
        for fn in sorted(os.listdir(PAPERS)):
            if fn.lower().endswith(".pdf"):
                m = ID.match(fn)
                (by_id.__setitem__(m.group(1), fn) if m else other.append(fn))
    return by_id, other


def write_manifest(by_id, other):
    with open(os.path.join(PAPERS, "MANIFEST.tsv"), "w") as f:
        f.write("arxiv_id\tfile\n")
        for aid, fn in sorted(by_id.items()):
            f.write(f"{aid}\t{fn}\n")
        for fn in other:
            f.write(f"\t{fn}\n")


def fetch():
    for line in open(os.path.join(PAPERS, "MANIFEST.tsv")).read().splitlines()[1:]:
        aid, fn = line.split("\t")
        dst = os.path.join(PAPERS, fn)
        if os.path.exists(dst):
            continue
        if not aid:
            print(f"skip (no arXiv ID, add by hand): {fn}")
            continue
        print(f"fetch {aid} -> {fn}")
        urllib.request.urlretrieve(f"https://arxiv.org/pdf/{aid}", dst)
        time.sleep(3)  # arXiv asks for at most one request every 3 s


def main():
    if "--fetch" in sys.argv:
        return fetch()
    works, order = load()
    by_id, other = local_pdfs()
    write_manifest(by_id, other)
    refs = {aid: i + 1 for i, e in enumerate(PAPER_REFS) for aid in (e if isinstance(e, tuple) else (e,)) if aid}
    md = ["# Paper index", "",
          "Generated by [build_index.py](build_index.py) from the bibliographies of the five research notes. "
          "Do not edit by hand.", "",
          f"{len(works)} cited works; {sum(1 for k in works if works[k]['id'] in by_id)} have a local PDF in "
          "[papers/](papers/) (not tracked in git; `python3 lit-review/build_index.py --fetch` restores them). "
          "**Ref** gives the work's number in the paper's reference list (`paper_scaffold.md`). "
          "**Also in** names the other notes that cite it.", ""]
    for th, name in THEMES.items():
        rows = [works[k] for k in order if works[k]["themes"][0] == th]
        rows.sort(key=lambda w: (w["year"] or "0", w["author"]))
        md += [f"## {th} · {name} ({len(rows)})", "",
               "| Year | First author | Title | arXiv | PDF | Ref | Also in |", "|---|---|---|---|---|---|---|"]
        for w in rows:
            link = f"[{w['id']}](https://arxiv.org/abs/{w['id']})" if w["id"] else "—"
            pdf = f"[pdf](papers/{by_id[w['id']]})" if w["id"] in by_id else ""
            ref = str(refs.get(w["id"], ""))
            also = ", ".join(w["themes"][1:])
            title = w["title"].replace("|", "/")
            md.append(f"| {w['year']} | {w['author']} | {title} | {link} | {pdf} | {ref} | {also} |")
        md.append("")
    cited = {works[k]["id"] for k in works}
    extra = [(aid, fn) for aid, fn in sorted(by_id.items()) if aid not in cited] + [("", fn) for fn in other]
    if extra:
        md += ["## Local PDFs not cited in the notes", "", "| arXiv | File | Ref |", "|---|---|---|"]
        for aid, fn in extra:
            link = f"[{aid}](https://arxiv.org/abs/{aid})" if aid else "—"
            md.append(f"| {link} | [{fn}](papers/{fn.replace(' ', '%20')}) | {refs.get(aid, '')} |")
        md.append("")
    missing = [aid for aid in refs if aid not in cited and aid not in by_id]
    if missing:
        md += ["Paper references with no entry above: " + ", ".join(
            f"[{a}](https://arxiv.org/abs/{a})" for a in missing) + ".", ""]
    open(os.path.join(HERE, "papers.md"), "w").write("\n".join(md))
    print(f"{len(works)} works, {len(by_id) + len(other)} local PDFs -> papers.md, papers/MANIFEST.tsv")


if __name__ == "__main__":
    main()
