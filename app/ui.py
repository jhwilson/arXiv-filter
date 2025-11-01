import os
import glob
import subprocess
from datetime import datetime
from typing import Tuple

import streamlit as st
import yaml
import re


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
RECS_GLOB = os.path.join(BASE_DIR, 'recommendations', 'recommendations_*.md')
WHITELIST_PATH = os.path.join(BASE_DIR, 'config', 'whitelist_authors.txt')


def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def list_recommendation_files():
    files = glob.glob(RECS_GLOB)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def parse_recommendation_sections(md_text: str) -> Tuple[str, str]:
    # Split by headers produced by the pipeline
    pri_hdr = '## Priority (whitelist authors)'
    oth_hdr = '## Other recommendations'
    pri_idx = md_text.find(pri_hdr)
    oth_idx = md_text.find(oth_hdr)

    if pri_idx != -1 and oth_idx != -1:
        pri = md_text[pri_idx + len(pri_hdr):oth_idx].strip()
        rest = md_text[oth_idx + len(oth_hdr):].strip()
        return pri, rest
    # Fallback: no sections → show everything as "Other"
    return '', md_text


def count_entries(md_text: str) -> int:
    # Count markdown entries as lines starting with level-3 headings
    return sum(1 for line in md_text.splitlines() if line.strip().startswith('### '))


def parse_entry_links(md_text: str):
    links = []
    for line in md_text.splitlines():
        s = line.strip()
        if s.startswith('### [') and '](' in s and s.endswith(')'):
            try:
                # Format: ### [Title](URL)
                title = s.split('[', 1)[1].split(']', 1)[0]
                url = s.split('](', 1)[1].rsplit(')', 1)[0]
                links.append((title, url))
            except Exception:
                continue
    return links


def split_entries(md_text: str):
    entries = []
    current = []
    started = False
    for line in md_text.splitlines():
        if line.startswith('### '):
            if started and current:
                entries.append('\n'.join(current).strip())
                current = []
            current = [line]
            started = True
            continue
        if not started:
            # ignore content before first entry title
            continue
        if line.strip() == '---':
            if current:
                entries.append('\n'.join(current).strip())
                current = []
            continue
        current.append(line)
    if started and current:
        entries.append('\n'.join(current).strip())
    return [e for e in entries if e]


def parse_entry(entry_md: str):
    title = ''
    url = ''
    authors = ''
    date = ''
    score = None
    abstract = ''

    lines = entry_md.splitlines()
    if lines and lines[0].startswith('### '):
        s = lines[0].strip()
        if s.startswith('### [') and '](' in s:
            try:
                title = s.split('[', 1)[1].split(']', 1)[0]
                url = s.split('](', 1)[1].rsplit(')', 1)[0]
            except Exception:
                pass

    # Gather fields
    for i, s in enumerate(lines):
        t = s.strip()
        if t.startswith('**Authors:**'):
            authors = t.replace('**Authors:**', '').strip()
        elif t.startswith('**Date:**'):
            date = t.replace('**Date:**', '').strip()
        elif t.startswith('**Similarity Score:**'):
            try:
                score = float(t.replace('**Similarity Score:**', '').strip())
            except Exception:
                score = None
        elif t.startswith('**Abstract:**'):
            # rest of lines after this
            abstract = '\n'.join(lines[i + 1:]).strip()
            break

    return {
        'title': title,
        'url': url,
        'authors': authors,
        'date': date,
        'score': score,
        'abstract': abstract,
    }


def sanitize_abstract(text: str) -> str:
    # Remove fenced code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Replace TeX-style quote patterns first: `word' -> 'word'
    text = re.sub(r"`([^'\n]+)'", r"'\1'", text)
    # Two backticks → double quote
    text = text.replace('``', '"')
    # Replace inline code spans `text` -> text
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Any remaining lone backticks → apostrophe
    text = text.replace('`', "'")
    return text


def score_to_color(score: float, threshold: float) -> str:
    if score is None:
        return '#cccccc'
    if score >= max(threshold + 0.1, threshold + 0.08):
        return '#22c55e'  # green-500
    if score >= threshold:
        return '#f59e0b'  # amber-500
    return '#ef4444'      # red-500


def _arxiv_id_from_url(url: str) -> str:
    if not url:
        return ''
    # pdf pattern
    m = re.search(r"/pdf/([^/]+)\.pdf", url)
    if m:
        return m.group(1)
    # abs pattern
    m = re.search(r"/abs/([^/?#]+)", url)
    if m:
        return m.group(1)
    # html pattern
    m = re.search(r"/html/([^/?#]+)", url)
    if m:
        return m.group(1)
    return ''


def render_card(item: dict, cfg: dict, whitelist: list = None, highlight: bool = False):
    title = item['title'] or 'Untitled'
    url = item['url'] or '#'
    authors = item['authors']
    date = item['date']
    score = item['score']
    threshold = float(cfg.get('similarity_threshold', 0.0))
    pct = max(0, min(100, int(round((score or 0.0) * 100))))
    color = score_to_color(score or 0.0, threshold)
    fill_pct = max(1, pct)  # ensure at least a sliver so the circle is visible

    # Build authors HTML (optionally highlighting whitelist matches)
    if highlight and whitelist:
        authors_html = authors_with_highlight(authors, whitelist)
    else:
        authors_html = f"<span class='rec-author'>{authors}</span>"

    # Build additional arXiv links (abs, html) next to title
    arx_id = _arxiv_id_from_url(url)
    extra_links_html = ''
    if arx_id:
        abs_url = f"https://arxiv.org/abs/{arx_id}"
        html_url = f"https://arxiv.org/html/{arx_id}"
        extra_links_html = f"<span style='margin-left:8px;font-size:12px;'>[<a href='{abs_url}' target='_blank'>abstract</a> · <a href='{html_url}' target='_blank'>html</a>]</span>"

    header_html = f"""
    <div style='display:flex;align-items:center;gap:12px;'>
      <div title='Similarity {pct}%' style='width:32px;height:32px;aspect-ratio:1/1;flex-shrink:0;border-radius:50%;
           background: conic-gradient({color} {fill_pct}%, var(--rec-muted) 0); border:2px solid var(--rec-border);'></div>
      <div>
        <a href='{url}' target='_blank' style='font-weight:700;font-size:16px;text-decoration:none;'>{title}</a>{extra_links_html}<br/>
        <div style='margin-top:4px;'>
          <span class='rec-label'>Authors: </span>
          {authors_html}
          <span class='rec-meta' style='margin-left:10px;'>Date: {date}</span>
          <span class='rec-meta' style='margin-left:10px;'>Score: {score if score is not None else 'N/A'}</span>
        </div>
      </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    if item['abstract']:
        # Sanitize backticks to avoid accidental code formatting, keep LaTeX intact
        st.markdown(sanitize_abstract(item['abstract']))
    st.markdown('---')


def load_whitelist_authors() -> list:
    try:
        with open(WHITELIST_PATH, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f if l.strip()]
    except Exception:
        return []


def _normalize_name(name: str) -> str:
    s = name.lower()
    for ch in [',', '.', '"', "'", '-', '_', '(', ')']:
        s = s.replace(ch, ' ')
    return ' '.join(s.split())


def _first_last(name: str):
    parts = _normalize_name(name).split()
    if not parts:
        return ('', '')
    if len(parts) == 1:
        return (parts[0], '')
    return (parts[0], parts[-1])


def _author_match(a: str, b: str) -> bool:
    af, al = _first_last(a)
    bf, bl = _first_last(b)
    if not al or not bl or al != bl:
        return False
    if af == bf:
        return True
    return af and bf and af[0] == bf[0]


def authors_with_highlight(authors_text: str, whitelist: list) -> str:
    # Return HTML string with whitelist matches highlighted
    parts = [p.strip() for p in authors_text.split(',') if p.strip()]
    styled = []
    for p in parts:
        is_wl = any(_author_match(p, w) for w in whitelist)
        if is_wl:
            styled.append(f"<span class='rec-author-hl'>{p}</span>")
        else:
            styled.append(f"<span class='rec-author'>{p}</span>")
    return ', '.join(styled)


def run_cmd(cmd, cwd=BASE_DIR) -> Tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
        out = (proc.stdout or '') + '\n' + (proc.stderr or '')
        return proc.returncode, out
    except Exception as e:
        return 1, str(e)


def _format_date_label(date_str: str) -> str:
    # date_str expected 'YYYY-MM-DD'
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
    except Exception:
        return date_str


def render_history(files) -> str:
    # Build descriptive labels and clickable list in the sidebar; return current selection path
    file_info = []
    for p in files:
        name = os.path.basename(p)
        date_str = name.replace('recommendations_', '').replace('.md', '')
        human = _format_date_label(date_str)
        try:
            with open(p, 'r', encoding='utf-8') as f:
                md = f.read()
            pri_md, oth_md = parse_recommendation_sections(md)
            pri_n = count_entries(pri_md)
            oth_n = count_entries(oth_md)
        except Exception:
            pri_n = 0
            oth_n = 0
        label = f"{human} — Priority {pri_n}, Other {oth_n}"
        file_info.append((p, label))

    if 'selected_rec_path' not in st.session_state:
        st.session_state.selected_rec_path = file_info[0][0]

    show_all = False
    if len(file_info) > 20:
        show_all = st.checkbox(f'Show all ({len(file_info)})', value=False)
    display_list = file_info if show_all else file_info[:20]

    for p, label in display_list:
        if st.button(label, key=f'sb_{label}', use_container_width=True):
            st.session_state.selected_rec_path = p
            st.experimental_rerun()

    return st.session_state.selected_rec_path


def ui_recommendations():
    st.title('ArXiv Recommendations')

    files = list_recommendation_files()
    if not files:
        st.info('No recommendations yet. Click "Run Pipeline" to generate.')
        return

    selected_path = st.session_state.get('selected_rec_path') or files[0]

    with open(selected_path, 'r', encoding='utf-8') as f:
        md = f.read()

    priority_md, other_md = parse_recommendation_sections(md)

    tab1, tab2 = st.tabs(["Priority", "Other recommendations"])
    with tab1:
        if priority_md.strip():
            pri_links = parse_entry_links(priority_md)
            col_content, col_select = st.columns([3, 1])
            with col_content:
                cfg = load_config()
                wl = load_whitelist_authors()
                pri_entries = [parse_entry(e) for e in split_entries(priority_md)]
                for it in pri_entries:
                    render_card(it, cfg, whitelist=wl, highlight=True)
            with col_select:
                st.subheader('Select')
                selected = []
                for idx, (title, url) in enumerate(pri_links):
                    checked = st.checkbox(title, key=f'pri_{selected_path}_{idx}')
                    if checked:
                        selected.append(url)
                if selected:
                    urls_text = '\n'.join(selected)
                    st.text_area('Links', value=urls_text, height=160)
                    st.download_button('Download (.txt)', data=urls_text, file_name='priority_links.txt', use_container_width=True)
        else:
            st.info('No priority recommendations in this result.')
    with tab2:
        if other_md.strip():
            oth_links = parse_entry_links(other_md)
            col_content, col_select = st.columns([3, 1])
            with col_content:
                cfg = load_config()
                oth_entries = [parse_entry(e) for e in split_entries(other_md)]
                for it in oth_entries:
                    render_card(it, cfg)
            with col_select:
                st.subheader('Select')
                selected = []
                for idx, (title, url) in enumerate(oth_links):
                    checked = st.checkbox(title, key=f'oth_{selected_path}_{idx}')
                    if checked:
                        selected.append(url)
                if selected:
                    urls_text = '\n'.join(selected)
                    st.text_area('Links', value=urls_text, height=160)
                    st.download_button('Download (.txt)', data=urls_text, file_name='other_links.txt', use_container_width=True)
        else:
            st.info('No other recommendations in this result.')

    st.divider()
    st.subheader('Actions')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Reload My Papers'):
            with st.spinner('Reloading my papers...'):
                # Fetch latest authored abstracts
                code, out = run_cmd(['python', 'src/fetch_abstracts.py'])
                st.code(out)
                if code == 0:
                    st.success('Reloaded my papers.')
                else:
                    st.error('Failed to reload my papers.')
    with col2:
        if st.button('Run Pipeline'):
            with st.spinner('Running pipeline... this may take a bit.'):
                code, out = run_cmd(['python', 'src/run_pipeline.py'])
                st.code(out)
                if code == 0:
                    st.success('Pipeline completed. Refresh the left panel to view the latest file.')
                else:
                    st.error('Pipeline failed. See output above.')


def ui_settings():
    st.title('Settings')
    cfg = load_config()

    st.caption('Edit key settings and click Save. Advanced edits can be done in config.yaml directly.')

    with st.form('settings_form'):
        default_author_id = st.text_input('Default Author ID', value=str(cfg.get('default_author_id', '')))
        days = st.number_input('Days window', min_value=0, max_value=30, step=1, value=int(cfg.get('days', 1)))

        categories = cfg.get('categories', []) or []
        categories_text = st.text_area('Categories (one per line)', value='\n'.join(categories), height=200)

        top_n = st.number_input('Top N', min_value=1, max_value=200, step=1, value=int(cfg.get('top_n', 20)))
        sim_thresh = st.number_input('Similarity threshold', min_value=0.0, max_value=1.0, step=0.01, value=float(cfg.get('similarity_threshold', 0.0)))

        embedding_model = st.text_input('Embedding model', value=str(cfg.get('embedding_model', 'allenai/specter2')))
        specter2_use_adapters = st.checkbox('Use SPECTER2 adapters mode', value=bool(cfg.get('specter2_use_adapters', False)))
        local_model_dir = st.text_input('Local model dir (single-repo mode)', value=str(cfg.get('local_model_dir', '')))
        local_specter2_base_dir = st.text_input('Local S2 base dir', value=str(cfg.get('local_specter2_base_dir', '')))
        local_specter2_adapter_dir = st.text_input('Local S2 adapter dir', value=str(cfg.get('local_specter2_adapter_dir', '')))

        submitted = st.form_submit_button('Save')
        if submitted:
            cfg['default_author_id'] = default_author_id
            cfg['days'] = int(days)
            cfg['categories'] = [c.strip() for c in categories_text.splitlines() if c.strip()]
            cfg['top_n'] = int(top_n)
            cfg['similarity_threshold'] = float(sim_thresh)
            cfg['embedding_model'] = embedding_model
            cfg['specter2_use_adapters'] = bool(specter2_use_adapters)
            cfg['local_model_dir'] = local_model_dir
            cfg['local_specter2_base_dir'] = local_specter2_base_dir
            cfg['local_specter2_adapter_dir'] = local_specter2_adapter_dir
            save_config(cfg)
            st.success('Settings saved.')


def main():
    st.set_page_config(page_title='ArXiv Filter', layout='wide')
    # Inject theme-aware CSS once
    if 'rec_css' not in st.session_state:
        st.session_state.rec_css = True
        st.markdown(
            """
            <style>
              :root {
                --rec-text: #111827; /* gray-900 */
                --rec-muted: #e5e7eb; /* gray-200 for circle background */
                --rec-border: #e5e7eb; /* circle border */
                --rec-meta: #6b7280; /* gray-500 */
                --rec-label: #374151; /* gray-700 */
                --rec-author: #111827; /* gray-900 */
                --rec-author-hl: #2563eb; /* blue-600 */
              }
              @media (prefers-color-scheme: dark) {
                :root {
                  --rec-text: #e5e7eb; /* gray-200 */
                  --rec-muted: #374151; /* gray-700 */
                  --rec-border: #374151; /* gray-700 */
                  --rec-meta: #9ca3af; /* gray-400 */
                  --rec-label: #d1d5db; /* gray-300 */
                  --rec-author: #e5e7eb; /* gray-200 */
                  --rec-author-hl: #60a5fa; /* blue-400 */
                }
              }
              .rec-meta { color: var(--rec-meta); font-size: 12px; }
              .rec-label { color: var(--rec-label); font-weight: 600; }
              .rec-author { color: var(--rec-author); font-size: 14px; font-weight: 600; }
              .rec-author-hl { color: var(--rec-author-hl); font-size: 14px; font-weight: 700; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    with st.sidebar:
        st.image('https://static-00.iconduck.com/assets.00/arxiv-icon-512x512-cb4l3jg5.png', width=32)
        view = st.radio('View', ['Recommendations', 'Settings'], index=0)
        if view == 'Recommendations':
            st.header('History')
            files = list_recommendation_files()
            if files:
                selected = render_history(files)
                st.session_state.selected_rec_path = selected

    if view == 'Recommendations':
        ui_recommendations()
    else:
        ui_settings()


if __name__ == '__main__':
    main()


