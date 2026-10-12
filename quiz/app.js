/* Paper quiz — UI. Content and scheduling live in engine.js; this file draws
 * questions, fits them to the screen, and handles editing and storage.
 *
 * Storage (localStorage, all under "phq1."):
 *   edits     {papers: {id: paper}, removed: [id]} — a paper you edit or add is
 *             stored whole and replaces the shipped one; untouched papers keep
 *             following the shipped data.
 *   progress  the engine's per-card memory, calibration and history.
 *   settings  target accuracy, filters, display options.
 *
 * No scrolling, no jitter: the page is a fixed three-row grid (bar, stage,
 * footer). Each question is laid out invisibly, its font size binary-searched
 * down until it fits the stage, and only then shown; if even the smallest size
 * overflows, the engine is asked for fewer options. Answering changes colours
 * only — captions under the options and every footer button hold their space
 * from the start (visibility, not display), so nothing moves.
 */
(function () {
  'use strict';
  const E = window.QuizEngine;
  const DATA = window.QUIZ_DATA || { version: 'none', papers: [] };
  const PREFIX = 'phq1.';
  const TIER = ['varied distractors', 'same-area distractors', 'close distractors'];
  const MIN_FS = 11.5;
  const LAST_FS = 10;     // only when two options will not fit at MIN_FS

  // ------------------------------------------------------------------ storage
  const store = {
    ok: true,
    get(k, d) {
      try {
        const v = localStorage.getItem(PREFIX + k);
        return v == null ? d : JSON.parse(v);
      } catch (e) { this.ok = false; return d; }
    },
    set(k, v) {
      try { localStorage.setItem(PREFIX + k, JSON.stringify(v)); return true; }
      catch (e) { this.ok = false; showSaveWarn(); return false; }
    },
  };

  function normEdits(e) {
    e = e || {};
    return { papers: e.papers || {}, base: e.base || {}, removed: e.removed || [] };
  }
  let edits = normEdits(store.get('edits', null));
  let progress = store.get('progress', null) || E.freshProgress();
  const defaults = E.defaultSettings();
  const savedSet = store.get('settings', {}) || {};
  let settings = Object.assign({}, defaults, savedSet, {
    types: Object.assign({}, defaults.types, savedSet.types || {}),
    areas: Object.assign({}, defaults.areas, savedSet.areas || {}),
  });
  settings.theme = settings.theme || 'auto';

  let index = E.buildIndex(E.mergePapers(DATA.papers, edits));
  const engine = new E.Engine(index, progress, settings);
  engine.set = settings;   // one settings object, shared
  const baseById = {};
  DATA.papers.forEach(p => { baseById[p.id] = p; });

  function saveProgress() { store.set('progress', progress); }
  function saveEdits() { store.set('edits', edits); }
  function saveSettings() { store.set('settings', settings); }
  function rebuild() { index = E.buildIndex(E.mergePapers(DATA.papers, edits)); engine.setIndex(index); }

  // ------------------------------------------------------------------ dom
  const $ = id => document.getElementById(id);
  const stage = $('stage'), verdict = $('verdict'), meta = $('meta'), statsEl = $('stats');
  const bNext = $('next'), bEdit = $('b-edit'), bThesis = $('b-thesis'), bMis = $('b-mis');
  const modal = $('modal');

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else if (k === 'value') el.value = v;
      else if (k === 'checked') el.checked = !!v;
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  function segsToNodes(segs) {
    return segs.map(s => s.blank ? h('span', { class: 'blank', title: 'blanked term' }) : document.createTextNode(s.t));
  }

  function showSaveWarn() { $('savewarn').hidden = store.ok; }

  function applyTheme() {
    if (settings.theme === 'light' || settings.theme === 'dark') document.documentElement.dataset.theme = settings.theme;
    else delete document.documentElement.dataset.theme;
  }

  // ------------------------------------------------------------------ question view
  let cur = null;   // {q, answered, chosen, res, snap}
  let advanceTimer = null;

  function paperOf(q) { return index.byId[q.card.pid]; }
  function titleBits(p) {
    return [h('span', { class: 'title' }, '‘' + p.title + '’'), settings.showYear && p.year ? ' (' + p.year + ')' : ''];
  }

  function promptEl(q) {
    const p = paperOf(q), c = q.card;
    if (c.type === 'a') return [h('div', { class: 'q' }, 'Who wrote ', ...titleBits(p), '?')];
    if (c.type === 't') return [h('div', { class: 'q' }, 'Which statement best captures the view of ', ...titleBits(p), '?')];
    if (c.type === 'd') {
      const t = E.findTerm(p, c.ref);
      return [h('div', { class: 'q' }, 'How does ', ...titleBits(p), ' define ', h('span', { class: 'term' }, '“' + t.term + '”'), '?')];
    }
    const x = E.findExample(p, c.ref);
    const segs = E.maskSegments(x.text, [x.phenomenon].concat(x.aliases || []));
    return [h('div', { class: 'q-sub' }, 'Which phenomenon does this example illustrate?'),
            h('div', { class: 'q example' }, ...segsToNodes(segs))];
  }

  function caption(q, o) {
    const p = index.byId[o.pid];
    const own = o.correct;
    if (o.kind === 'author') return own ? '' : (p ? '‘' + p.title + '’' : '');
    if (o.kind === 'thesis') {
      if (!p) return '';
      if (own) return 'this paper';
      return E.shortAuthors(p.authors) + ', ‘' + p.title + '’' + (p.year ? ' (' + p.year + ')' : '');
    }
    if (o.kind === 'def') {
      const t = p && E.findTerm(p, o.tid);
      if (!t) return '';
      return '“' + t.term + '” — ' + (own ? 'this paper' : E.shortAuthors(p.authors) + ', ‘' + p.title + '’');
    }
    return '';
  }

  function renderQuestion(q) {
    stage.textContent = '';
    stage.classList.remove('revealed', 'intro', 'enter');
    if (q.intro) stage.classList.add('intro');
    const kick = h('div', { class: 'kicker' }, h('span', {}, E.TYPE_LABEL[q.card.type]));
    if (q.intro) kick.append(h('span', { class: 'badge' }, 'New'));
    else kick.append(h('span', {}, q.k + ' options'), h('span', {}, '· ' + TIER[q.tier]));
    stage.append(kick, ...promptEl(q));
    const ol = h('ol', { id: 'opts' });
    const withCap = q.card.type !== 'x' && !q.intro;
    q.options.forEach((o, i) => {
      const content = E.optionContent(index, o);
      const short = o.kind === 'author' || o.kind === 'phen';
      const btn = h('button', { class: 'opt' + (short ? ' short' : ''), 'data-i': i, onclick: e => choose(i, e) },
        h('span', { class: 'key' }, String(i + 1)),
        h('span', { class: 'body' },
          h('span', { class: 'txt' }, ...segsToNodes(content.segs)),
          withCap ? h('span', { class: 'cap' }, caption(q, o)) : null));
      ol.append(h('li', {}, btn));
    });
    stage.append(ol);
  }

  function fits() { return stage.scrollHeight <= stage.clientHeight + 1 && stage.scrollWidth <= stage.clientWidth + 1; }
  function maxFs() { return window.innerWidth < 560 ? 18 : 21; }

  /* Largest font size (floor..max) at which the stage does not overflow. */
  function fit(floor) {
    let lo = floor || MIN_FS, hi = maxFs();
    stage.style.setProperty('--fs', hi + 'px');
    if (fits()) return true;
    stage.style.setProperty('--fs', lo + 'px');
    if (!fits()) return false;
    for (let i = 0; i < 7; i++) {
      const mid = (lo + hi) / 2;
      stage.style.setProperty('--fs', mid + 'px');
      if (fits()) lo = mid; else hi = mid;
    }
    stage.style.setProperty('--fs', lo + 'px');
    return true;
  }

  /* Fit at the normal floor, else at the last-resort one. */
  function refit() { return fit() || fit(LAST_FS); }

  /* Lay out q invisibly, fit it, show it. A redraw (keepState) keeps the
   * answered state and skips the fade. False if it overflows even at LAST_FS. */
  function present(q, keepState) {
    stage.style.visibility = 'hidden';
    renderQuestion(q);
    const ok = refit();
    if (keepState && cur && cur.answered) applyReveal();
    stage.style.visibility = '';
    void stage.offsetWidth;
    if (!keepState) {
      stage.classList.add('enter');
      if (cur) cur.shownAt = performance.now();
    }
    return ok;
  }

  /* Plan `card` and show it with as many options as fit: start at kCap and
   * ask the engine for one fewer each time the layout overflows. */
  function presentCard(card, now, kCap) {
    for (kCap = kCap || E.MODEL.kMax; ; kCap--) {
      const q = engine.plan(card, now, kCap);
      if (!q) return false;                  // engine passes over it for this visit
      cur = { q, answered: false };
      if (present(q) || q.k <= 2) return true;
    }
  }

  function nextQuestion() {
    clearTimeout(advanceTimer);
    const now = Date.now();
    for (let tries = 0; tries < 50; tries++) {
      const card = engine.pickCard(now);
      if (!card) { showEmpty(); return; }
      if (presentCard(card, now)) { setFooterAsked(); updateStats(); return; }
    }
    showEmpty();
  }

  /* After a resize or a redraw that no longer fits: an unanswered question
   * is re-planned with fewer options; an answered one keeps its options. */
  function refitCurrent() {
    if (!cur) return;
    if (refit() || cur.answered || cur.q.intro || cur.q.k <= 2) return;
    presentCard(cur.q.card, Date.now(), cur.q.k - 1);
    setFooterAsked();
  }

  function showEmpty() {
    cur = null;
    stage.textContent = '';
    stage.append(h('div', { class: 'empty' }, index.cards.length
      ? 'Nothing to ask with the current filters. Open Settings to turn some question types or areas back on.'
      : 'No papers loaded.'));
    verdict.textContent = ''; meta.textContent = '';
    bNext.disabled = true; bEdit.disabled = true;
    bThesis.classList.add('slot-hide'); bMis.classList.add('slot-hide');
    updateStats();
  }

  function setFooterAsked() {
    const q = cur.q;
    bEdit.disabled = false;
    bMis.classList.add('slot-hide');
    bThesis.classList.toggle('slot-hide', q.card.type !== 't');
    verdict.className = '';
    if (q.intro) {
      verdict.textContent = 'New card';
      meta.textContent = 'Read it, then tap the answer (or press Enter) to continue';
      bNext.disabled = false;
    } else {
      verdict.textContent = '';
      meta.textContent = 'Choose one · keys 1–' + q.k;
      bNext.disabled = true;
    }
  }

  function metaText(q) {
    const p = paperOf(q), c = q.card;
    if (!p) return '';
    if (c.type === 'a') return E.authorString(p.authors) + ' · ' + (p.topic || '');
    if (c.type === 't') {
      const th = p.theses[p.pick] || {};
      return (th.kind === 'excerpt' ? 'Excerpt' : 'Summary') + ' ' + ((p.pick | 0) + 1) + ' of ' + p.theses.length +
        ' · ' + E.shortAuthors(p.authors) + ' · “Thesis…” to pick another';
    }
    if (c.type === 'd') {
      const t = E.findTerm(p, c.ref) || {};
      return (t.exact ? 'Exact excerpt' : 'Paraphrase') + ' · ' + E.shortAuthors(p.authors) + ' ' + (p.year || '');
    }
    const x = E.findExample(p, c.ref) || {};
    return 'From ' + E.shortAuthors(p.authors) + ', ‘' + p.title + '’' + (p.year ? ' (' + p.year + ')' : '') +
      ' · ' + (x.exact ? 'exact excerpt' : 'paraphrase');
  }

  function applyReveal() {
    const q = cur.q;
    stage.classList.add('revealed');
    stage.querySelectorAll('.opt').forEach((b, i) => {
      b.disabled = true;
      b.classList.toggle('correct', !!q.options[i].correct);
      b.classList.toggle('wrong', i === cur.chosen && !q.options[i].correct);
    });
    const ok = cur.res.correct;
    verdict.className = ok ? 'ok' : 'bad';
    verdict.textContent = cur.forced ? '✓ Counted as correct' : (ok ? '✓ Correct' : '✗ Not quite');
    verdict.append(h('span', { class: 'exp' }, 'expected ' + Math.round(q.p * 100) + '%'));
    meta.textContent = metaText(q);
    bNext.disabled = false;
    bMis.classList.toggle('slot-hide', ok);
    bThesis.classList.toggle('slot-hide', q.card.type !== 't');
  }

  /* A pointer click within moments of a question appearing is the second
   * half of a double-click (or double-tap) aimed at the previous one. Keyboard
   * activation has detail 0 and is never ignored. */
  function tooSoon(e) { return !!(e && e.detail && cur && cur.shownAt && performance.now() - cur.shownAt < 350); }

  function choose(i, e) {
    if (!cur || cur.answered || modalOpen() || tooSoon(e)) return;
    const q = cur.q;
    if (i < 0 || i >= q.options.length) return;
    const now = Date.now();
    if (q.intro) {
      engine.answer(q, true, now);
      saveProgress();
      nextQuestion();
      return;
    }
    cur.snap = engine.snapshot(q.card);
    cur.res = engine.answer(q, !!q.options[i].correct, now);
    cur.answered = true;
    cur.chosen = i;
    cur.at = now;
    saveProgress();
    applyReveal();
    updateStats();
    if (cur.res.correct && settings.autoAdvance) advanceTimer = setTimeout(nextQuestion, 1300);
  }

  function misclick() {
    if (!cur || !cur.answered || cur.res.correct) return;
    engine.restore(cur.q.card, cur.snap);
    cur.res = engine.answer(cur.q, true, cur.at || Date.now());
    cur.forced = true;
    saveProgress();
    applyReveal();
    updateStats();
  }

  function onNext(e) {
    if (!cur || modalOpen() || tooSoon(e)) return;
    if (cur.q.intro && !cur.answered) { choose(cur.q.options.findIndex(o => o.correct)); return; }
    if (cur.answered) nextQuestion();
  }

  function updateStats() {
    const s = engine.stats(Date.now());
    statsEl.textContent = '';
    const acc = s.acc == null ? '–' : Math.round(s.acc * 100) + '%';
    const wide = window.innerWidth >= 560;
    statsEl.append(h('b', {}, acc), wide ? ' of last ' + (s.recent || 0) : '', ' · ', h('b', {}, String(s.seen)),
      '/' + s.total + (wide ? ' seen' : ''), ' · ', h('b', {}, String(s.due)), ' due');
    statsEl.title = 'Accuracy over your last ' + s.recent + ' answers (target ' + Math.round(settings.target * 100) +
      '%); cards seen of all active cards; cards due for review';
    showSaveWarn();
  }

  // ------------------------------------------------------------------ modal shell
  let modalCtx = null;   // {onKey, onClose, canClose}
  function modalOpen() { return !modal.hidden; }
  /* Focus goes back to the control that opened a dialog only if you reached
   * it from the keyboard (Tab). After a mouse click or a shortcut it is left
   * on the page instead, because a button given focus back right after a
   * keypress counts as :focus-visible and would take the next Enter (meant
   * for "Next") — reopening the dialog you just closed. */
  let lastFocus = null;
  function openModal(panel, ctx) {
    clearTimeout(advanceTimer);
    if (modal.hidden) {
      const a = document.activeElement;
      lastFocus = a && a !== document.body && a.matches(':focus-visible') ? a : null;
    }
    modal.textContent = '';
    modal.append(panel);
    modal.hidden = false;
    $('app').inert = true;              // nothing behind the dialog can be focused or clicked
    panel.tabIndex = -1;
    panel.focus({ preventScroll: true });
    modalCtx = ctx || {};
  }
  /* Close the modal; unless forced, a dialog with unsaved work may ask first. */
  function closeModal(force) {
    if (force !== true && modalCtx && modalCtx.canClose && !modalCtx.canClose()) return;
    modal.hidden = true;
    modal.textContent = '';
    $('app').inert = false;
    const ctx = modalCtx; modalCtx = null;
    if (ctx && ctx.onClose) ctx.onClose();
    if (!modal.hidden) return;
    if (lastFocus && document.contains(lastFocus)) lastFocus.focus({ preventScroll: true });
    else if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
    lastFocus = null;
  }
  modal.addEventListener('mousedown', e => { if (e.target === modal) closeModal(); });

  function panelShell(title, { tabs, foot }) {
    const body = h('div', { class: 'p-body' });
    const head = h('div', { class: 'p-headwrap' },
      h('div', { class: 'p-head' }, h('h2', {}, title), h('button', { class: 'x', title: 'Close (Esc)', onclick: () => closeModal() }, '×')),
      tabs || null);
    const panel = h('div', { class: 'panel', role: 'dialog', 'aria-label': title }, head, body, h('div', { class: 'p-foot' }, foot || []));
    return { panel, body };
  }

  // ------------------------------------------------------------------ editor
  const clone = o => JSON.parse(JSON.stringify(o));
  const stable = E.stable;
  const uid = () => 'u' + Date.now().toString(36) + Math.floor(Math.random() * 1e4).toString(36);
  const topics = () => [...new Set(index.papers.map(p => p.topic).filter(Boolean))].sort();

  function blankPaper() {
    return {
      id: 'u-' + Date.now().toString(36), title: '', authors: [], year: '', area: 'language', topic: '', formal: false,
      theses: [{ kind: 'excerpt', text: '' }, { kind: 'excerpt', text: '' }, { kind: 'summary', text: '' }, { kind: 'summary', text: '' }],
      pick: 0, terms: [], examples: [],
    };
  }

  /* `back`, if given, reopens whatever the editor was opened from. */
  function openEditor(pid, tab, ref, back) {
    const existing = pid && index.byId[pid];
    const paper = existing ? clone(existing) : blankPaper();
    delete paper._edited; delete paper._added;
    paper.theses = paper.theses || []; paper.terms = paper.terms || []; paper.examples = paper.examples || [];
    const id = paper.id;
    const off = {};
    const offKey = (type, r) => id + '|' + type + (r ? '|' + r : '');
    const isOff = (type, r) => (offKey(type, r) in off) ? off[offKey(type, r)] : !!progress.off[offKey(type, r)];
    const setOff = (type, r, v) => { off[offKey(type, r)] = v; };
    const ed = {
      tab: tab || 'paper',
      ti: Math.max(0, paper.terms.findIndex(t => t.id === ref)),
      xi: Math.max(0, paper.examples.findIndex(x => x.id === ref)),
    };
    const isBase = !!baseById[id];
    const tabsEl = h('div', { class: 'tabs' });
    const TABS = [['paper', 'Paper'], ['thesis', 'Thesis'], ['terms', 'Terms'], ['examples', 'Examples']];
    const footLeft = h('div', { class: 'left' });
    const shell = panelShell(existing ? 'Edit — ' + (paper.title || 'paper') : 'New paper', {
      tabs: tabsEl,
      foot: [footLeft, h('button', { class: 'btn', onclick: () => closeModal() }, 'Cancel'),
             h('button', { class: 'btn primary', onclick: save }, 'Save')],
    });
    const body = shell.body;

    if (isBase && edits.papers[id]) {
      footLeft.append(h('button', { class: 'btn', title: 'Discard your edits to this paper', onclick: () => {
        if (!confirm('Discard your edits and restore the shipped version of this paper?')) return;
        delete edits.papers[id]; delete edits.base[id]; saveEdits(); rebuild(); closeModal(true); refreshAfterEdit();
      } }, 'Restore original'));
    }
    if (existing) {
      footLeft.append(h('button', { class: 'btn danger', onclick: () => {
        if (!confirm(isBase ? 'Hide this paper from the quiz? (Settings can bring hidden papers back.)' : 'Delete this paper?')) return;
        if (isBase) { if (!edits.removed.includes(id)) edits.removed.push(id); }
        else { delete edits.papers[id]; delete edits.base[id]; }
        saveEdits(); rebuild(); closeModal(true); refreshAfterEdit();
      } }, isBase ? 'Hide paper' : 'Delete paper'));
    }

    function drawTabs() {
      tabsEl.textContent = '';
      for (const [k, label] of TABS) {
        const n = k === 'terms' ? ' (' + paper.terms.length + ')' : k === 'examples' ? ' (' + paper.examples.length + ')' : '';
        tabsEl.append(h('button', { class: 'tab' + (ed.tab === k ? ' on' : ''), onclick: () => { ed.tab = k; draw(); } }, label + n));
      }
    }

    const field = (label, input, cls) => h('label', { class: 'f' + (cls ? ' ' + cls : '') }, label, input);
    const text = (val, onInput, attrs) => h('input', Object.assign({ type: 'text', value: val || '', oninput: e => onInput(e.target.value) }, attrs || {}));
    const area = (val, onInput, attrs) => h('textarea', Object.assign({ oninput: e => onInput(e.target.value) }, attrs || {}), val || '');
    const check = (label, val, onChange, title) => h('label', { class: 'check', title },
      h('input', { type: 'checkbox', checked: val, onchange: e => onChange(e.target.checked) }), label);

    function drawPaper() {
      const dl = h('datalist', { id: 'topic-list' }, topics().map(t => h('option', { value: t })));
      body.append(
        field('Title', text(paper.title, v => { paper.title = v; })),
        h('div', { class: 'row' },
          field('Authors (separate with ;)', text((paper.authors || []).join('; '), v => { paper.authors = v.split(';').map(s => s.trim()).filter(Boolean); })),
          h('div', { class: 'narrow', style: 'width:7em' }, field('Year', text(paper.year, v => { paper.year = v.trim(); })))),
        h('div', { class: 'row' },
          field('Area', h('select', { onchange: e => { paper.area = e.target.value; } },
            E.AREAS.map(a => h('option', { value: a, selected: paper.area === a ? 'selected' : null }, E.AREA_LABEL[a])))),
          field('Topic (wrong options come from the same topic)', text(paper.topic, v => { paper.topic = v; }, { list: 'topic-list' }))),
        dl,
        h('div', { class: 'row', style: 'flex-wrap:wrap; gap:16px' },
          h('div', { class: 'narrow' }, check('Formal / semi-formal', !!paper.formal, v => { paper.formal = v; })),
          h('div', { class: 'narrow' }, check('Quiz me on the author', !isOff('a'), v => setOff('a', null, !v))),
          h('div', { class: 'narrow' }, check('Quiz me on the thesis', !isOff('t'), v => setOff('t', null, !v)))),
        h('div', { class: 'hint' }, isBase ? (edits.papers[id] ? 'Shipped paper — you have edited it.' : 'Shipped paper.')
          : 'Added by you.', ' Edits are saved in this browser only; use Settings → Export to back them up.'));
    }

    function drawThesis() {
      body.append(h('div', { class: 'hint' }, 'The quiz uses the selected statement. Excerpts should be verbatim and at most 30 words; summaries are in your words.'));
      paper.theses.forEach((t, i) => {
        const row = h('div', { class: 'cand' + (paper.pick === i ? ' picked' : '') },
          h('div', { class: 'cand-side' },
            h('label', { class: 'check', title: 'Quiz me on this one' },
              h('input', { type: 'radio', name: 'pick', checked: paper.pick === i, onchange: () => { paper.pick = i; draw(); } }), 'use'),
            h('select', { onchange: e => { t.kind = e.target.value; } },
              h('option', { value: 'excerpt', selected: t.kind === 'excerpt' ? 'selected' : null }, 'excerpt'),
              h('option', { value: 'summary', selected: t.kind !== 'excerpt' ? 'selected' : null }, 'summary')),
            h('button', { class: 'x', style: 'font-size:15px', title: 'Remove this candidate', onclick: () => {
              paper.theses.splice(i, 1);
              if (paper.pick >= paper.theses.length) paper.pick = Math.max(0, paper.theses.length - 1);
              else if (paper.pick > i) paper.pick--;
              draw();
            } }, '×')),
          area(t.text, v => { t.text = v; }, { 'aria-label': 'Candidate ' + (i + 1) }));
        body.append(row);
      });
      if (paper.theses.length < 6) {
        body.append(h('div', {}, h('button', { class: 'btn', onclick: () => { paper.theses.push({ kind: 'summary', text: '' }); draw(); } }, '+ Candidate')));
      }
    }

    function navBar(kind, list, idx, setIdx, make) {
      return h('div', { class: 'nav' },
        h('button', { class: 'btn', disabled: idx <= 0, onclick: () => { setIdx(idx - 1); draw(); } }, '‹'),
        h('span', {}, list.length ? (kind + ' ' + (idx + 1) + ' of ' + list.length) : 'No ' + kind.toLowerCase() + 's yet'),
        h('button', { class: 'btn', disabled: idx >= list.length - 1, onclick: () => { setIdx(idx + 1); draw(); } }, '›'),
        h('span', { class: 'sp' }),
        h('button', { class: 'btn', onclick: () => { list.push(make()); setIdx(list.length - 1); draw(); } }, '+ Add'),
        h('button', { class: 'btn danger', disabled: !list.length, onclick: () => {
          if (!confirm('Delete this ' + kind.toLowerCase() + '?')) return;
          list.splice(idx, 1); setIdx(Math.max(0, Math.min(idx, list.length - 1))); draw();
        } }, 'Delete'));
    }

    function drawTerms() {
      const list = paper.terms;
      body.append(navBar('Term', list, ed.ti, v => { ed.ti = v; }, () => ({ id: uid(), term: '', aliases: [], definition: '', exact: false })));
      const t = list[ed.ti];
      if (!t) { body.append(h('div', { class: 'hint' }, 'Add a term this paper introduces or defines.')); return; }
      body.append(
        h('div', { class: 'row' },
          field('Term', text(t.term, v => { t.term = v; })),
          field('Other forms to blank out (comma-separated)', text((t.aliases || []).join(', '), v => { t.aliases = v.split(',').map(s => s.trim()).filter(Boolean); }))),
        field('How this paper defines it', area(t.definition, v => { t.definition = v; }), 'grow'),
        h('div', { class: 'row', style: 'gap:16px' },
          h('div', { class: 'narrow' }, check('Exact excerpt', !!t.exact, v => { t.exact = v; })),
          h('div', { class: 'narrow' }, check('Quiz me on this', !isOff('d', t.id), v => setOff('d', t.id, !v)))),
        h('div', { class: 'hint' }, 'The term and its other forms are blanked out wherever this definition is shown.'));
    }

    function drawExamples() {
      const list = paper.examples;
      body.append(navBar('Example', list, ed.xi, v => { ed.xi = v; },
        () => ({ id: uid(), text: '', phenomenon: '', aliases: [], near: [], avoid: [], exact: false })));
      const x = list[ed.xi];
      if (!x) { body.append(h('div', { class: 'hint' }, 'Add an example or scenario from this paper.')); return; }
      body.append(
        field('Example or scenario', area(x.text, v => { x.text = v; }), 'grow'),
        h('div', { class: 'row' },
          field('Phenomenon it illustrates', text(x.phenomenon, v => { x.phenomenon = v; })),
          field('Words to blank out of the example (comma-separated)', text((x.aliases || []).join(', '), v => { x.aliases = v.split(',').map(s => s.trim()).filter(Boolean); }))),
        field('Near misses — close but wrong answers (separate with ;)', text((x.near || []).join('; '), v => { x.near = v.split(';').map(s => s.trim()).filter(Boolean); })),
        field('Never offer as a wrong answer — names that are also right (separate with ;)', text((x.avoid || []).join('; '), v => { x.avoid = v.split(';').map(s => s.trim()).filter(Boolean); })),
        h('div', { class: 'row', style: 'gap:16px' },
          h('div', { class: 'narrow' }, check('Exact excerpt', !!x.exact, v => { x.exact = v; })),
          h('div', { class: 'narrow' }, check('Quiz me on this', !isOff('x', x.id), v => setOff('x', x.id, !v)))));
    }

    function draw() {
      drawTabs();
      body.textContent = '';
      ({ paper: drawPaper, thesis: drawThesis, terms: drawTerms, examples: drawExamples })[ed.tab]();
    }

    function save() {
      paper.title = (paper.title || '').trim();
      if (!paper.title) { alert('A paper needs a title.'); ed.tab = 'paper'; draw(); return; }
      if (!paper.authors || !paper.authors.length) { alert('A paper needs at least one author.'); ed.tab = 'paper'; draw(); return; }
      const keep = paper.theses.map((t, i) => [t, i]).filter(([t]) => (t.text || '').trim());
      const pickedIdx = keep.findIndex(([, i]) => i === paper.pick);
      paper.theses = keep.map(([t]) => Object.assign(t, { text: t.text.trim() }));
      paper.pick = pickedIdx >= 0 ? pickedIdx : 0;
      // A half-filled term or example is a mistake to point out, not to drop;
      // only one left entirely empty (an unused "+ Add") goes quietly.
      const filled = v => !!(v || '').trim();
      const ti = paper.terms.findIndex(t => filled(t.term) !== filled(t.definition));
      if (ti >= 0) {
        alert('Term ' + (ti + 1) + ' needs ' + (filled(paper.terms[ti].term) ? 'its definition.' : 'the term itself.'));
        ed.tab = 'terms'; ed.ti = ti; draw(); return;
      }
      const xi = paper.examples.findIndex(x => filled(x.text) !== filled(x.phenomenon));
      if (xi >= 0) {
        alert('Example ' + (xi + 1) + ' needs ' + (filled(paper.examples[xi].text) ? 'the phenomenon it illustrates.' : 'its text.'));
        ed.tab = 'examples'; ed.xi = xi; draw(); return;
      }
      paper.terms = paper.terms.filter(t => filled(t.term));
      paper.examples = paper.examples.filter(x => filled(x.text));
      for (const [k, v] of Object.entries(off)) { if (v) progress.off[k] = 1; else delete progress.off[k]; }
      if (isBase && stable(paper) === stable(baseById[id])) { delete edits.papers[id]; delete edits.base[id]; }
      else {
        edits.papers[id] = paper;
        if (isBase) edits.base[id] = clone(baseById[id]);   // what the edit was made from, for merging later fixes
      }
      edits.removed = edits.removed.filter(r => r !== id);
      saveEdits(); saveProgress(); rebuild(); closeModal(true); refreshAfterEdit();
    }

    const initial = stable(paper);
    const dirty = () => stable(paper) !== initial || Object.entries(off).some(([k, v]) => v !== !!progress.off[k]);
    openModal(shell.panel, {
      onKey: e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { save(); return true; } return false; },
      canClose: () => !dirty() || confirm('Discard your unsaved changes to this paper?'),
      onClose: back || null,
    });
    draw();
  }

  /* After an edit: redraw the current question from the new content, keeping
   * its options and answered state; move on if its card no longer exists. */
  function refreshAfterEdit() {
    if (!cur) { nextQuestion(); return; }
    const card = index.cardById[cur.q.card.id];
    const alive = card && engine.active(card) && cur.q.options.every(o => !o.pid || index.byId[o.pid]);
    if (!alive) { nextQuestion(); return; }
    if (!cur.answered && !cur.q.intro && optionsStale()) {
      if (!presentCard(card, Date.now())) { nextQuestion(); return; }
      setFooterAsked(); updateStats(); return;
    }
    cur.q.card = card;
    if (cur.q.card.type === 't' || cur.q.card.type === 'a' || cur.q.card.type === 'x') refreshCorrect();
    if (!present(cur.q, true)) refitCurrent();
    if (!cur.answered) setFooterAsked();
    updateStats();
  }
  /* An unanswered question is stale when the edit has ruled out one of its
   * wrong options (now an avoid entry, now the same as the answer, deleted,
   * or emptied): it is planned again for the same card. */
  function optionsStale() {
    const fresh = E.candidates(index, cur.q.card);
    if (!fresh) return true;
    const key = o => o.kind === 'thesis' ? 't:' + o.pid : o.kind === 'def' ? 'd:' + o.pid + '/' + o.tid
      : o.kind + ':' + E.norm(o.text);
    const valid = new Set(fresh.tiers.flat().map(key));
    const right = E.norm((E.optionContent(index, fresh.correct).segs || []).map(x => x.t).join(''));
    return cur.q.options.some(o => {
      if (o.correct) return false;
      const text = E.norm((E.optionContent(index, o).segs || []).map(x => x.t).join(''));
      return !valid.has(key(o)) || !text || text === right;
    });
  }
  // author and phenomenon options carry their text; refresh the right one from the content
  function refreshCorrect() {
    const fresh = E.candidates(index, cur.q.card);
    if (!fresh) return;
    cur.q.options = cur.q.options.map(o => o.correct ? Object.assign({ correct: true }, fresh.correct) : o);
  }

  // ------------------------------------------------------------------ library
  const libState = { query: '', page: 0 };
  function openLibrary() {
    const st = libState;
    const listEl = h('div', { class: 'lib-list' });
    const pageInfo = h('span', { class: 'hint' });
    const prev = h('button', { class: 'btn', onclick: () => { st.page--; drawList(); } }, '‹ Prev');
    const next = h('button', { class: 'btn', onclick: () => { st.page++; drawList(); } }, 'Next ›');
    const shell = panelShell('Library — ' + index.papers.length + ' papers', {
      foot: [h('div', { class: 'left' }, h('button', { class: 'btn', onclick: () => openEditor(null, 'paper', null, openLibrary) }, '+ New paper')),
             prev, pageInfo, next],
    });
    const search = h('input', { type: 'search', value: st.query, placeholder: 'Search title, author, topic, term…', oninput: e => { st.query = e.target.value; st.page = 0; drawList(); } });
    shell.body.append(search, listEl);
    const seenBy = {};
    for (const c of index.cards) {
      seenBy[c.pid] = seenBy[c.pid] || [0, 0];
      seenBy[c.pid][1]++;
      if (progress.cards[c.id]) seenBy[c.pid][0]++;
    }
    function matches(p, q) {
      if (!q) return true;
      const hay = E.norm([p.title, (p.authors || []).join(' '), p.year, p.topic, E.AREA_LABEL[p.area],
        (p.terms || []).map(t => t.term).join(' '), (p.examples || []).map(x => x.phenomenon).join(' ')].join(' '));
      return E.norm(q).split(' ').every(w => hay.includes(w));
    }
    let per = 0;
    function drawList() {
      const all = index.papers.filter(p => matches(p, st.query));
      const first = st.page * per;
      const was = per;
      per = Math.max(3, Math.floor(listEl.clientHeight / 46));
      if (was && was !== per) st.page = Math.floor(first / per);
      const pages = Math.max(1, Math.ceil(all.length / per));
      st.page = Math.max(0, Math.min(st.page, pages - 1));
      listEl.textContent = '';
      for (const p of all.slice(st.page * per, st.page * per + per)) {
        const [seen, n] = seenBy[p.id] || [0, 0];
        listEl.append(h('button', { class: 'lib-row', onclick: () => openEditor(p.id, 'paper', null, openLibrary) },
          h('span', { class: 't' }, p.title),
          h('span', { class: 'm' }, [E.shortAuthors(p.authors), p.year, E.AREA_LABEL[p.area], p.topic,
            seen + '/' + n + ' cards seen', p._added ? 'added by you' : p._edited ? 'edited' : ''].filter(Boolean).join(' · '))));
      }
      if (!all.length) listEl.append(h('div', { class: 'hint' }, 'No papers match.'));
      pageInfo.textContent = (st.page + 1) + ' / ' + pages + ' · ' + all.length + ' papers';
      prev.disabled = st.page <= 0;
      next.disabled = st.page >= pages - 1;
    }
    openModal(shell.panel, { onResize: drawList });
    drawList();
    search.focus();
  }

  // ------------------------------------------------------------------ settings
  function openSettings() {
    // All four sections at once when they fit; otherwise one at a time under tabs.
    const tabsEl = h('div', { class: 'tabs' });
    const shell = panelShell('Settings', { tabs: tabsEl, foot: [h('button', { class: 'btn primary', onclick: () => closeModal() }, 'Done')] });
    let tabbed = false, section = 0;
    const before = JSON.stringify([settings.types, settings.areas]);
    const changed = () => { saveSettings(); engine.set = settings; applyTheme(); draw(); };
    function chipRow(map, keys, labels) {
      return h('div', { class: 'chips' }, keys.map(k => h('button', {
        class: 'chip' + (map[k] !== false ? ' on' : ''),
        onclick: () => { map[k] = map[k] === false; if (!Object.values(map).some(v => v !== false)) map[k] = true; changed(); },
      }, labels[k])));
    }
    function draw() {
      const s = engine.stats(Date.now());
      const hidden = edits.removed.length;
      const body = shell.body;
      body.textContent = '';
      const grid = h('div', { class: 'set-grid' + (tabbed ? ' tabbed' : '') },
        h('div', {},
          h('h3', {}, 'Difficulty'),
          h('label', { class: 'f' }, 'Target accuracy: ' + Math.round(settings.target * 100) + '%',
            h('input', { type: 'range', min: 60, max: 95, step: 1, value: Math.round(settings.target * 100),
              onchange: e => { settings.target = (+e.target.value) / 100; changed(); } })),
          h('label', { class: 'f' }, 'Most options per question: ' + settings.maxOptions,
            h('input', { type: 'range', min: 2, max: 6, step: 1, value: settings.maxOptions,
              onchange: e => { settings.maxOptions = +e.target.value; changed(); } })),
          h('div', { class: 'chips', style: 'margin-top:8px' },
            h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: settings.showYear, onchange: e => { settings.showYear = e.target.checked; changed(); } }), 'Show year with title'),
            h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: settings.autoAdvance, onchange: e => { settings.autoAdvance = e.target.checked; changed(); } }), 'Auto-advance when right')),
          h('label', { class: 'f', style: 'margin-top:8px' }, 'Theme',
            h('select', { onchange: e => { settings.theme = e.target.value; changed(); } },
              ['auto', 'light', 'dark'].map(t => h('option', { value: t, selected: settings.theme === t ? 'selected' : null }, t))))),
        h('div', {},
          h('h3', {}, 'Question types'),
          chipRow(settings.types, E.TYPES, { a: 'Author', t: 'Thesis', d: 'Definition', x: 'Example' }),
          h('h3', { style: 'margin-top:12px' }, 'Areas'),
          chipRow(settings.areas, E.AREAS, E.AREA_LABEL)),
        h('div', {},
          h('h3', {}, 'Progress'),
          h('div', { class: 'kv' },
            h('div', {}, h('b', {}, String(s.seen)), ' of ', h('b', {}, String(s.total)), ' active cards seen; ',
              h('b', {}, String(s.solid)), ' remembered for a day or more'),
            h('div', {}, h('b', {}, String(s.due)), ' due now · ', h('b', {}, String(progress.q)), ' answers in all'),
            h('div', {}, 'Last ', h('b', {}, String(s.recent)), ': ', h('b', {}, s.acc == null ? '–' : Math.round(s.acc * 100) + '%'), ' right'))),
        h('div', {},
          h('h3', {}, 'Data (this browser only)'),
          h('div', { class: 'chips' },
            h('button', { class: 'btn', onclick: exportData }, 'Export'),
            h('button', { class: 'btn', onclick: importData }, 'Import'),
            hidden ? h('button', { class: 'btn', onclick: () => { edits.removed = []; saveEdits(); rebuild(); draw(); } }, 'Unhide ' + hidden + ' paper' + (hidden > 1 ? 's' : '')) : null,
            h('button', { class: 'btn danger', onclick: () => {
              if (!confirm('Forget all progress? Your edits, and the cards you switched off, are kept.')) return;
              const fresh = E.freshProgress();
              fresh.off = progress.off || {};
              if (!store.set('progress', fresh)) { alert('This browser is not letting the page save, so nothing was reset.'); return; }
              location.reload();
            } }, 'Reset progress'),
            h('button', { class: 'btn danger', onclick: () => {
              if (!confirm('Discard every edit and added paper? Progress is kept.')) return;
              if (!store.set('edits', normEdits(null))) { alert('This browser is not letting the page save, so nothing was discarded.'); return; }
              location.reload();
            } }, 'Discard edits'))));
      body.append(grid);
      const names = ['Difficulty', 'Filters', 'Progress', 'Data'];
      tabsEl.textContent = '';
      tabsEl.hidden = !tabbed;
      if (tabbed) {
        [...grid.children].forEach((el, i) => { el.hidden = i !== section; });
        names.forEach((n, i) => tabsEl.append(h('button', { class: 'tab' + (i === section ? ' on' : ''), onclick: () => { section = i; draw(); } }, n)));
      } else if (body.scrollHeight > body.clientHeight + 1) {
        tabbed = true;
        draw();
      }
    }
    openModal(shell.panel, { onClose: () => {
      const after = JSON.stringify([settings.types, settings.areas]);
      if (!cur || (after !== before && !(engine.active(cur.q.card)))) nextQuestion();
      else if (cur) { if (!present(cur.q, true)) refitCurrent(); if (!cur.answered) setFooterAsked(); updateStats(); }
    } });
    draw();
  }

  function exportData() {
    const blob = new Blob([JSON.stringify({ app: 'paper-quiz', v: 1, exported: new Date().toISOString(),
      data: DATA.version, edits, progress, settings }, null, 1)], { type: 'application/json' });
    const a = h('a', { href: URL.createObjectURL(blob), download: 'paper-quiz-' + new Date().toISOString().slice(0, 10) + '.json' });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  function importData() {
    const input = h('input', { type: 'file', accept: 'application/json,.json' });
    input.addEventListener('change', () => {
      const f = input.files && input.files[0];
      if (!f) return;
      f.text().then(txt => {
        let d;
        try { d = JSON.parse(txt); } catch (e) { alert('That file is not valid JSON.'); return; }
        if (!d || d.app !== 'paper-quiz' || !d.progress || !d.edits) { alert('That is not a paper-quiz export.'); return; }
        if (!confirm('Replace this browser\'s progress, edits and settings with the file\'s?')) return;
        const ok = store.set('edits', normEdits(d.edits)) && store.set('progress', d.progress) &&
          store.set('settings', d.settings || {});
        if (!ok) {
          store.set('edits', edits); store.set('progress', progress); store.set('settings', settings);
          alert('This browser is not letting the page save, so the import was not applied.');
          return;
        }
        location.reload();
      });
    });
    input.click();
  }

  // ------------------------------------------------------------------ other tabs
  // Each tab holds progress and edits in memory and writes them whole, so a
  // tab left open would overwrite what another tab saved. The storage event
  // fires in every other tab on each write: take the new state in there.
  window.addEventListener('storage', e => {
    if (!e.key || !e.key.startsWith(PREFIX)) return;
    const k = e.key.slice(PREFIX.length);
    if (k === 'progress') {
      progress = store.get('progress', null) || E.freshProgress();
      engine.prog = progress;
      for (const t of E.TYPES) if (typeof progress.beta[t] !== 'number') progress.beta[t] = 0;
      updateStats();
    } else if (k === 'edits') {
      edits = normEdits(store.get('edits', null));
      rebuild();
      if (!modalOpen()) refreshAfterEdit();
    } else if (k === 'settings') {
      const v = store.get('settings', {}) || {};
      Object.assign(settings, v, { types: Object.assign({}, defaults.types, v.types || {}),
        areas: Object.assign({}, defaults.areas, v.areas || {}) });
      applyTheme();
      updateStats();
    }
  });

  // ------------------------------------------------------------------ wiring
  bNext.addEventListener('click', e => onNext(e));
  bMis.addEventListener('click', misclick);
  bEdit.addEventListener('click', () => {
    if (!cur) return;
    const c = cur.q.card;
    openEditor(c.pid, { a: 'paper', t: 'thesis', d: 'terms', x: 'examples' }[c.type], c.ref);
  });
  bThesis.addEventListener('click', () => { if (cur) openEditor(cur.q.card.pid, 'thesis'); });
  $('b-lib').addEventListener('click', openLibrary);
  $('b-set').addEventListener('click', openSettings);

  document.addEventListener('keydown', e => {
    if (modalOpen()) {
      if (e.key === 'Escape') { e.preventDefault(); closeModal(); return; }
      if (modalCtx && modalCtx.onKey && modalCtx.onKey(e)) e.preventDefault();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = (e.target && e.target.tagName) || '';
    if (/INPUT|TEXTAREA|SELECT/.test(tag)) return;
    const focused = document.activeElement;
    if (/^[1-9]$/.test(e.key)) { e.preventDefault(); choose(+e.key - 1); }
    else if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowRight') {
      // A button reached with Tab (focus-visible) does what it says; a button
      // that merely kept focus after a mouse click does not steal Enter.
      if (e.key !== 'ArrowRight' && focused && focused.tagName === 'BUTTON' && focused !== bNext &&
          !focused.disabled && focused.matches(':focus-visible')) return;
      e.preventDefault(); onNext();
    }
    else if (e.key === 'm' || e.key === 'M') { e.preventDefault(); misclick(); }
    else if (e.key === 't' || e.key === 'T') { e.preventDefault(); if (cur && cur.q.card.type === 't') openEditor(cur.q.card.pid, 'thesis'); }
    else if (e.key === 'e' || e.key === 'E') { e.preventDefault(); bEdit.click(); }
    else if (e.key === 'l' || e.key === 'L') { e.preventDefault(); openLibrary(); }
    else if (e.key === 's' || e.key === 'S') { e.preventDefault(); openSettings(); }
  });

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      refitCurrent();
      updateStats();
      if (modalCtx && modalCtx.onResize) modalCtx.onResize();
    }, 80);
  });

  applyTheme();
  nextQuestion();
})();
