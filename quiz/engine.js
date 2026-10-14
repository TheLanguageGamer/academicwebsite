/* Paper quiz — content model and adaptive engine (no DOM; runs in Node for tests).
 *
 * Cards. Each paper yields an author card (a), a thesis card (t), one card per
 * term (d, "definition") and one per example (x). A card's id is built from the
 * paper id and the term/example id, so edits to the wording keep its history.
 *
 * Model. For card c shown with k options whose distractors have mean closeness
 * cost D_s, after Δt minutes since it was last seen:
 *
 *     P(correct) = 1/k + (1 − 1/k) · σ(θ_c + α + β_type − γ·ln(k−1) − D_s − F(Δt/h_c))
 *     F(r) = F_max · (1 − (1 + r)^(−1/2))
 *
 * F is forgetting: about 0.9 logits at one half-life, levelling off at F_max.
 * It is bounded because this is recognition among options, not recall — a
 * paper you once knew does not sink to chance — and an unbounded curve made
 * every long-overdue card look forgotten (predictions ran 15 points low).
 *
 * θ_c is how well you discriminate this card's answer, h_c its memory
 * half-life (minutes), α how strong you are overall and β_type a
 * per-question-type offset. Each answer's evidence is shared out between
 * θ_c, α and β_type in proportion to how unsure each still is (a Laplace,
 * Kalman-style step on the likelihood): at first nothing is known about you,
 * so a run of right answers raises α quickly and every card gets harder
 * within a dozen answers; as the estimates firm up the steps shrink, and a
 * little process noise keeps α and β free to follow you as you learn. h_c
 * grows on a success (more for a harder, better-spaced success) and shrinks
 * on a miss.
 *
 * The three difficulty levers the quiz pulls:
 *   - number of options k (2–6; 1 for a card's first showing — a lesson —
 *     when even its easiest question is likely beyond you, or always if new
 *     cards are set to start as lessons);
 *   - closeness of the wrong options (tier 0 other areas, 1 same area, 2 same
 *     topic, 3 the same paper or author, same-named terms, hand-picked near
 *     misses);
 *   - spacing: a card comes due once the time since you last saw it reaches
 *     τ·h_c. τ is 1 at target and moves with your results (down to 0.3 when
 *     you are below target, so reviews come sooner; up to 1.6 above it).
 * Each turn takes the most overdue due card (else a new card — of the next
 * one of each type, the one pitched nearest the aim; or, while you are
 * persistently below target, a nearly due card first — else the card nearest
 * to due), passing over any too easy to ask, and the (k, closeness) whose
 * predicted accuracy is nearest
 * the aim. The aim is the target (80% by default) nudged by a running average
 * of recent results (proportional) and by their accumulated shortfall or
 * excess (integral), so a bad patch eases off, a good one tightens up, and
 * whatever the model gets wrong about you is corrected by your results.
 *
 * Why spacing runs on h and not on predicted accuracy: θ can always be met
 * with a harder presentation, so "due when predicted accuracy drops" makes a
 * moderately known card due forever. Memory decay is what spacing is for.
 */
(function (root, factory) {
  const E = factory();
  if (typeof module === 'object' && module.exports) module.exports = E;
  else root.QuizEngine = E;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const TYPES = ['a', 't', 'd', 'x'];
  const TYPE_LABEL = { a: 'Author', t: 'Thesis', d: 'Definition', x: 'Example' };
  const AREAS = ['language', 'epistemology', 'action', 'rationality', 'metaphysics', 'logic'];
  const AREA_LABEL = {
    language: 'Philosophy of language', epistemology: 'Epistemology', action: 'Action',
    rationality: 'Reasons & rationality', metaphysics: 'Metaphysics', logic: 'Logic',
  };

  const M = {
    gamma: 0.55,            // cost per ln(k−1) beyond the guessing floor
    delta: [0, 0.9, 1.8, 2.7],   // cost of a distractor by closeness tier
    fMax: 3.0,              // forgetting ceiling (logits)
    th0: 1.5,               // θ of a card just introduced
    h0: 2,                  // half-life (min) of a card just introduced
    hKnown: 60,             // ... of one you got right the first time it was asked (before the success step)
    hMin: 1.5,
    hMax: 60 * 24 * 365 * 2,
    tauGain: 2.5, tauMin: 0.3, tauMax: 1.6, // spacing multiplier around 1
    // prior variances (logit²) and per-answer process noise of the estimates
    alpha0: 1.5,            // α to start from: the quiz is for a reader who knows the field;
                            // a few answers correct it either way
    va0: 2.0, qa: 0.004,    // α: you, over all question types
    vb0: 0.5, qb: 0.002,    // β: one question type relative to α
    vt0: 1.0, vtMin: 0.25,  // θ: one card (kept above vtMin, since you go on learning it)
    vn0: 1.0, qn: 0.002,    // ν: cards not yet asked, relative to those you have been
    gap: 4,                 // a card is not repeated within this many questions
    passMargin: 0.06, passMax: 8,   // next(): pass over reviews this far above the aim (at most this many)
    lessonMargin: 0.15,     // a new card predicted this far below the aim at its easiest is shown as a lesson
    kMax: 6,
    ewmaRate: 0.08,
    aimGain: 0.5, aimSpan: 0.08,          // proportional term
    ctlRate: 0.012, ctlMin: -0.3, ctlMax: 0.15,   // integral term
  };
  const DMAX = M.gamma * Math.log(M.kMax - 1) + M.delta[M.delta.length - 1];

  const sigmoid = x => 1 / (1 + Math.exp(-x));
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

  // ------------------------------------------------------------- text helpers

  function norm(s) {
    return String(s || '').normalize('NFKD').replace(/[̀-ͯ]/g, '')
      .toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  }

  function authorString(authors) {
    const a = (authors || []).map(x => String(x).trim()).filter(Boolean);
    if (a.length <= 1) return a[0] || '';
    if (a.length === 2) return a[0] + ' & ' + a[1];
    return a.slice(0, -1).join(', ') + ' & ' + a[a.length - 1];
  }

  function surname(name) {
    const parts = String(name || '').trim().split(/\s+/).filter(p => !/^(jr\.?|ii|iii)$/i.test(p));
    let k = parts.length - 1;
    while (k > 0 && /^(van|von|de|der|den)$/i.test(parts[k - 1])) k--;
    return parts.slice(k).join(' ');
  }

  /* Identity of a person across spellings: "David Lewis" = "David K. Lewis". */
  function personKey(name) {
    const sur = norm(surname(name)).replace(/ /g, '');
    const first = norm(String(name || '').trim().split(/\s+/)[0] || '').charAt(0);
    return sur + ':' + first;
  }
  function authorsKey(authors) { return (authors || []).map(personKey).sort().join('|'); }

  function shortAuthors(authors) {
    const s = (authors || []).map(surname).filter(Boolean);
    if (s.length <= 2) return s.join(' & ');
    return s[0] + ' et al.';
  }

  /* Word sets are asked for again and again (every definition against every
   * other, on every question), so they are kept by string. The strings are
   * the content's own, so the caches stay as small as the content. */
  function memo(fn) {
    const cache = new Map();
    return s => {
      s = String(s || '');
      let v = cache.get(s);
      if (v === undefined) cache.set(s, v = fn(s));
      return v;
    };
  }
  const words = memo(s => new Set(norm(s).split(' ').filter(w => w.length > 2)));
  function jaccard(a, b) {
    const A = words(a), B = words(b);
    if (!A.size || !B.size) return 0;
    let n = 0;
    for (const w of A) if (B.has(w)) n++;
    return n / (A.size + B.size - n);
  }

  /* Title words that a right answer repeats ("common ground" in the statement
   * picked for 'Common Ground', "essential indexical" for 'The Problem of the
   * Essential Indexical'). Every question names the paper, so it could be
   * answered by matching words; plan() makes sure at least one wrong answer
   * repeats one of them too. */
  const CUE_STOP = new Set(['that', 'this', 'with', 'from', 'what', 'when', 'which', 'some', 'about',
    'into', 'their', 'there', 'toward', 'towards', 'does', 'have']);
  // words compared by their first 7 letters, so "indexicality" repeats "indexical"
  const cueWords = memo(s => new Set(norm(s).split(' ').filter(w => w.length > 3 && !CUE_STOP.has(w))
    .map(w => stem(w).slice(0, 7))));
  function titleCue(title, text) {
    const t = cueWords(text);
    return [...cueWords(title)].filter(w => t.has(w));
  }
  function cueCount(cue, text) {
    if (!cue.length) return 0;
    const t = cueWords(text);
    return cue.filter(w => t.has(w)).length;
  }

  const escRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  /* Keys for spotting one label twice: its words in any order, and the same
   * without a parenthetical gloss — "homonymy (lexical ambiguity)" is
   * "lexical ambiguity (homonymy)", and "cross-sentential anaphora" is
   * "cross-sentential (discourse) anaphora". */
  const labelKeys = memo(s => {
    const words = t => labelTokens(t).slice().sort().join(' ');
    return [words(s), words(s.replace(/\([^)]*\)/g, ' '))].filter(Boolean);
  });
  /* A label's words with articles dropped and plurals reduced, so "the
   * essential indexical" is "essential indexical" and "paradoxes of material
   * implication" is "paradox of material implication". Deliberately crude:
   * it only has to make two spellings of one label agree. */
  const ARTICLES = new Set(['the', 'a', 'an']);
  function stem(w) {
    if (w.length <= 3) return w;
    if (/(x|z|ch|sh|ss)es$/.test(w)) return w.slice(0, -2);
    if (/[^siu]s$/.test(w)) return w.slice(0, -1);
    return w;
  }
  const labelTokens = memo(t => norm(t).split(' ').filter(w => w && !ARTICLES.has(w)).map(stem));

  /* Key-order-independent serialization, for "is this the same content?" */
  function stable(o) {
    if (Array.isArray(o)) return '[' + o.map(stable).join(',') + ']';
    if (o && typeof o === 'object') return '{' + Object.keys(o).filter(k => !k.startsWith('_')).sort()
      .map(k => JSON.stringify(k) + ':' + stable(o[k])).join(',') + '}';
    return JSON.stringify(o === undefined ? null : o);
  }

  /* Split `text` into [{t: string, blank: bool}] with every needle (and its
   * plural) blanked, whole-word and case-insensitive. */
  function maskSegments(text, needles) {
    const list = [...new Set((needles || []).map(n => String(n || '').trim()).filter(n => n.length > 1))]
      .sort((a, b) => b.length - a.length);
    text = String(text || '');
    if (!list.length) return [{ t: text, blank: false }];
    // The word boundary before a needle is matched (group 1) rather than
    // looked behind for: WebKit before 16.4 rejects lookbehind outright.
    const re = new RegExp('(^|[^\\p{L}\\p{N}])(?:' + list.map(escRe).join('|') +
      ')(?:s|es)?(?![\\p{L}\\p{N}])', 'giu');
    const out = [];
    let last = 0, m;
    while ((m = re.exec(text))) {
      const start = m.index + m[1].length, end = m.index + m[0].length;
      if (start > last) out.push({ t: text.slice(last, start), blank: false });
      out.push({ t: text.slice(start, end), blank: true });
      last = end;
      if (end === m.index) re.lastIndex++;
    }
    if (last < text.length) out.push({ t: text.slice(last), blank: false });
    return out;
  }

  // ------------------------------------------------------------- content index

  /* Merge shipped papers with the user's edits. An edit is stored as the
   * user's copy of the paper (edits.papers[id]) beside the shipped paper it
   * was made from (edits.base[id]), and merged three ways with the paper as
   * shipped now: whatever the user changed is kept, whatever they left alone
   * follows later shipped fixes — a corrected quotation, a new term, a new
   * `avoid` entry. Fields are merged one by one, terms and examples item by
   * item (by id); a pick moves with its statement when the candidates change.
   * An edit saved before bases were kept replaces the paper wholesale, as it
   * always did. A removed id drops the paper. */
  function merge3(old, mine, cur) {
    const same = (a, b) => stable(a) === stable(b);
    const out = {};
    const keys = new Set([...Object.keys(old), ...Object.keys(mine), ...Object.keys(cur)]);
    for (const k of keys) {
      if (k.startsWith('_') || ['theses', 'pick', 'terms', 'examples'].includes(k)) continue;
      const v = same(mine[k], old[k]) ? cur[k] : mine[k];
      if (v !== undefined) out[k] = v;
    }
    if (same(mine.theses, old.theses)) {
      out.theses = cur.theses || [];
      if (mine.pick === old.pick) out.pick = cur.pick;
      else {
        const chosen = ((old.theses || [])[mine.pick] || {}).text;
        const at = out.theses.findIndex(t => t.text === chosen);
        out.pick = at >= 0 ? at : clamp(mine.pick | 0, 0, Math.max(0, out.theses.length - 1));
      }
    } else {
      out.theses = mine.theses || [];
      out.pick = mine.pick;
    }
    for (const list of ['terms', 'examples']) {
      const byId = arr => new Map((arr || []).filter(x => x && x.id).map(x => [x.id, x]));
      const o = byId(old[list]), m = byId(mine[list]), c = byId(cur[list]);
      const merged = [];
      for (const [id, ci] of c) {
        if (o.has(id) && !m.has(id)) continue;                    // the user deleted it
        if (m.has(id) && !(o.has(id) && same(m.get(id), o.get(id)))) merged.push(m.get(id));  // the user changed it
        else merged.push(ci);                                      // untouched, or new in the shipped data
      }
      for (const [id, mi] of m) {
        if (c.has(id)) continue;
        if (o.has(id) && same(mi, o.get(id))) continue;           // the shipped data dropped it, untouched
        merged.push(mi);                                           // the user's own, or changed and kept
      }
      out[list] = merged;
    }
    return out;
  }

  function mergePapers(base, edits) {
    const e = edits || {};
    const over = e.papers || {};
    const bases = e.base || {};
    const removed = new Set(e.removed || []);
    const out = [];
    const seen = new Set();
    for (const p of base) {
      if (removed.has(p.id)) continue;
      const mine = over[p.id];
      if (!mine) out.push(p);
      else if (!bases[p.id]) out.push(Object.assign({}, mine, { _edited: true }));
      else out.push(Object.assign(merge3(bases[p.id], mine, p), { _edited: true,
        _baseChanged: stable(bases[p.id]) !== stable(p) }));
      seen.add(p.id);
    }
    for (const id of Object.keys(over)) {
      if (!seen.has(id) && !removed.has(id)) out.push(Object.assign({}, over[id], { _added: true, _edited: true }));
    }
    return out;
  }

  function thesisText(p) {
    const th = p.theses || [];
    const i = clamp(p.pick | 0, 0, Math.max(0, th.length - 1));
    return th[i] ? th[i].text : '';
  }

  /* Cards, pools and the order new cards are introduced in. */
  function buildIndex(papers) {
    const byId = {};
    papers.forEach(p => { byId[p.id] = p; });

    // Introduction order: philosophy of language and the rest interleaved 2:1
    // in shipped order, user-added papers first. A paper's author and thesis
    // cards come with it; its terms and examples a couple of papers later.
    const added = papers.filter(p => p._added);
    const lang = papers.filter(p => !p._added && p.area === 'language');
    const rest = papers.filter(p => !p._added && p.area !== 'language');
    const order = [...added];
    let i = 0, j = 0;
    while (i < lang.length || j < rest.length) {
      if (i < lang.length) order.push(lang[i++]);
      if (i < lang.length) order.push(lang[i++]);
      if (j < rest.length) order.push(rest[j++]);
    }

    const cards = [];
    order.forEach((p, n) => {
      const base = { pid: p.id, area: p.area, topic: p.topic };
      if (p.authors && p.authors.length) cards.push(Object.assign({ id: p.id + '|a', type: 'a', pri: n }, base));
      if (thesisText(p)) cards.push(Object.assign({ id: p.id + '|t', type: 't', pri: n + 0.5 }, base));
      (p.terms || []).forEach((t, m) => {
        if (t && t.id && t.term && t.definition)
          cards.push(Object.assign({ id: p.id + '|d|' + t.id, type: 'd', ref: t.id, pri: n + 2.2 + 0.7 * m }, base));
      });
      (p.examples || []).forEach((x, m) => {
        if (x && x.id && x.text && x.phenomenon)
          cards.push(Object.assign({ id: p.id + '|x|' + x.id, type: 'x', ref: x.id, pri: n + 2.5 + 0.7 * m }, base));
      });
    });
    cards.sort((a, b) => a.pri - b.pri);
    const cardById = {};
    cards.forEach(c => { cardById[c.id] = c; });

    // Option pools
    const authorPool = new Map();   // authorsKey -> {text, pids, topics, areas, years, names}
    const phenPool = new Map();     // norm -> {text, topics, areas, pids}
    papers.forEach(p => {
      const a = authorString(p.authors);
      if (a) {
        const k = authorsKey(p.authors);
        if (!authorPool.has(k)) authorPool.set(k, { text: a, pids: [], topics: new Set(), areas: new Set(), years: [], names: new Set() });
        const e = authorPool.get(k);
        e.pids.push(p.id); e.topics.add(p.topic); e.areas.add(p.area);
        e.years.push(parseInt(p.year, 10) || 0);
        (p.authors || []).forEach(n => e.names.add(personKey(n)));
      }
      (p.examples || []).forEach(x => {
        if (!x || !x.phenomenon) return;
        const k = norm(x.phenomenon);
        if (!phenPool.has(k)) phenPool.set(k, { text: x.phenomenon, topics: new Set(), areas: new Set(), pids: new Set() });
        const e = phenPool.get(k);
        e.topics.add(p.topic); e.areas.add(p.area); e.pids.add(p.id);
      });
    });
    return { papers, byId, cards, cardById, authorPool, phenPool };
  }

  function findTerm(p, ref) { return (p.terms || []).find(t => t.id === ref); }
  function findExample(p, ref) { return (p.examples || []).find(x => x.id === ref); }

  /* The text of an option reference, for display. Definitions come back as
   * mask segments (their own term blanked). */
  function optionContent(index, o) {
    const p = index.byId[o.pid];
    switch (o.kind) {
      case 'author': return { segs: [{ t: o.text, blank: false }] };
      case 'thesis': return { segs: [{ t: p ? thesisText(p) : '', blank: false }] };
      case 'def': {
        const t = p && findTerm(p, o.tid);
        if (!t) return { segs: [{ t: '', blank: false }] };
        return { segs: maskSegments(t.definition, [t.term].concat(t.aliases || [])) };
      }
      case 'phen': return { segs: [{ t: o.text, blank: false }] };
    }
    return { segs: [] };
  }

  /* Correct answer and tiered distractor candidates for a card. */
  function candidates(index, card) {
    const p = index.byId[card.pid];
    // 0 other areas, 1 same area, 2 same topic, 3 closest: the same
    // paper or author, a same-named term, a hand-picked near miss
    const tiers = [[], [], [], []];
    if (!p) return null;
    const area = p.area, topic = p.topic;

    if (card.type === 'a') {
      const text = authorString(p.authors);
      const key = authorsKey(p.authors);
      const myNames = new Set((p.authors || []).map(personKey));
      const year = parseInt(p.year, 10) || 0;
      for (const [k, e] of index.authorPool) {
        if (k === key) continue;
        const o = { kind: 'author', text: e.text, pid: e.pids[0] };
        const shares = [...e.names].some(n => myNames.has(n));
        const nearYear = e.years.some(y => Math.abs(y - year) <= 6);
        if (shares) tiers[3].push(o);
        else if (e.topics.has(topic) || (e.areas.has(area) && nearYear)) tiers[2].push(o);
        else if (e.areas.has(area)) tiers[1].push(o);
        else tiers[0].push(o);
      }
      return { correct: { kind: 'author', text, pid: p.id }, tiers };
    }

    if (card.type === 't') {
      const text = thesisText(p);
      const first = norm(surname((p.authors || [])[0]));
      // `avoid` names papers whose statements could pass for this one's
      const avoid = new Set(p.avoid || []);
      const cue = titleCue(p.title, text);
      for (const q of index.papers) {
        if (q.id === p.id || avoid.has(q.id) || (q.avoid || []).includes(p.id)) continue;
        const qt = thesisText(q);
        if (!qt || jaccard(qt, text) > 0.6) continue;
        const o = { kind: 'thesis', pid: q.id, cue: cueCount(cue, qt) };
        const sameAuthor = first && norm(surname((q.authors || [])[0])) === first;
        if (sameAuthor) tiers[3].push(o);
        else if (q.topic === topic) tiers[2].push(o);
        else if (q.area === area) tiers[1].push(o);
        else tiers[0].push(o);
      }
      return { correct: { kind: 'thesis', pid: p.id }, tiers, cue: cue.length > 0 };
    }

    if (card.type === 'd') {
      const t = findTerm(p, card.ref);
      if (!t) return null;
      const tn = norm(t.term);
      // `avoid` ("paper/term" refs) names definitions of the same notion that
      // could pass for this paper's; either side's list rules the pair out
      const me = p.id + '/' + t.id;
      const avoid = new Set(t.avoid || []);
      // the words a definition shows: its own, less its term's (which are
      // blanked) — set arithmetic, since masking all of them is too slow here
      const shown = u => {
        const hide = cueWords([u.term].concat(u.aliases || []).join(' '));
        return [...cueWords(u.definition)].filter(w => !hide.has(w));
      };
      const cue = (() => { const tw = cueWords(p.title); return shown(t).filter(w => tw.has(w)); })();
      for (const q of index.papers) {
        for (const u of q.terms || []) {
          if (!u || !u.definition || !u.term) continue;
          if (q.id === p.id && u.id === t.id) continue;
          if (avoid.has(q.id + '/' + u.id) || (u.avoid || []).includes(me)) continue;
          if (jaccard(u.definition, t.definition) > 0.6) continue;
          const o = { kind: 'def', pid: q.id, tid: u.id, cue: cue.length ? shown(u).filter(w => cue.includes(w)).length : 0 };
          if (q.id === p.id || norm(u.term) === tn) tiers[3].push(o);
          else if (q.topic === topic) tiers[2].push(o);
          else if (q.area === area) tiers[1].push(o);
          else tiers[0].push(o);
        }
      }
      return { correct: { kind: 'def', pid: p.id, tid: t.id }, tiers, cue: cue.length > 0 };
    }

    if (card.type === 'x') {
      const x = findExample(p, card.ref);
      if (!x) return null;
      const right = norm(x.phenomenon);
      // the question names the paper, whose title can share words with the answer
      const cue = titleCue(p.title, x.phenomenon);
      // `avoid` lists names that would also be a right answer for this example
      const avoid = new Set((x.avoid || []).flatMap(labelKeys));
      const seen = new Set(labelKeys(x.phenomenon));
      const repeat = n => labelKeys(n).some(k => seen.has(k) || avoid.has(k));
      const take = n => labelKeys(n).forEach(k => seen.add(k));
      // The example's own near misses were checked against it: offer each
      // unless it is the answer again or on the avoid list — even "non-natural
      // meaning" against "natural meaning", which is the point of the pair.
      for (const n of x.near || []) {
        if (!norm(n) || repeat(n)) continue;
        take(n);
        tiers[3].push({ kind: 'phen', text: n, cue: cueCount(cue, n) });
      }
      // Names from other papers were not: never one from the same topic (it
      // may be another name for the same thing), nor one that contains the
      // answer or is contained in it ("implicature" for "scalar implicature").
      const sameTopic = new Set();
      for (const [k, e] of index.phenPool) if (e.topics.has(topic)) sameTopic.add(k);
      const rightT = ' ' + labelTokens(x.phenomenon).join(' ') + ' ';
      const nests = name => {
        const t = ' ' + labelTokens(name).join(' ') + ' ';
        return t.trim() && (t.includes(rightT) || rightT.includes(t));
      };
      for (const [k, e] of index.phenPool) {
        if (!k || sameTopic.has(k) || nests(e.text) || repeat(e.text)) continue;
        take(e.text);
        const o = { kind: 'phen', text: e.text, cue: cueCount(cue, e.text) };
        if (e.areas.has(area)) tiers[1].push(o);
        else tiers[0].push(o);
      }
      return { correct: { kind: 'phen', text: x.phenomenon }, tiers, cue: cue.length > 0 };
    }
    return null;
  }

  /* Mean closeness cost of n distractors drawn from tier s first, then the
   * easier tiers, then (if the pool is that thin) the harder ones. */
  function composition(counts, n, s) {
    const order = [];
    for (let t = s; t >= 0; t--) order.push(t);
    for (let t = s + 1; t < counts.length; t++) order.push(t);
    let left = n, cost = 0, took = [];
    for (const t of order) {
      const m = Math.min(left, counts[t]);
      cost += m * M.delta[t];
      if (m) took.push([t, m]);
      left -= m;
      if (!left) break;
    }
    return { mean: n ? cost / (n - left || 1) : 0, short: left, took };
  }

  function shuffle(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  // ------------------------------------------------------------- engine

  function freshProgress() {
    return {
      v: 1, q: 0, cards: {}, beta: { a: 0, t: 0, d: 0, x: 0 },
      alpha: M.alpha0, va: M.va0, vb: { a: M.vb0, t: M.vb0, d: M.vb0, x: M.vb0 }, nu: 0, vn: M.vn0,
      ewma: null, ctl: 0, recent: [], hist: [], off: {},
    };
  }

  function defaultSettings() {
    return {
      target: 0.8, maxOptions: 6, showYear: true, advance: true, lessonFirst: false,
      types: { a: true, t: true, d: true, x: true },
      areas: Object.fromEntries(AREAS.map(a => [a, true])),
    };
  }

  class Engine {
    constructor(index, progress, settings, rng) {
      this.index = index;
      this.prog = progress || freshProgress();
      this.set = Object.assign(defaultSettings(), settings || {});
      this.rng = rng || Math.random;
      this.skip = new Set();
      this.setProgress(this.prog);
    }

    /* Adopt saved progress, filling in what older saves lack. */
    setProgress(prog) {
      this.prog = prog;
      if (!prog.beta) prog.beta = {};
      for (const t of TYPES) if (typeof prog.beta[t] !== 'number') prog.beta[t] = 0;
      if (typeof prog.alpha !== 'number') this.warmStart();
      if (!prog.vb) prog.vb = {};
      for (const t of TYPES) if (typeof prog.vb[t] !== 'number') prog.vb[t] = M.vb0;
      if (typeof prog.va !== 'number') prog.va = M.va0;
      if (typeof prog.nu !== 'number') { prog.nu = 0; prog.vn = M.vn0; }
    }

    /* A save from before α existed: estimate α from the answer history. Each
     * entry keeps k and the chance predicted at the time, so the logit of
     * the time can be recovered and the answers replayed. */
    warmStart() {
      const prog = this.prog;
      prog.alpha = M.alpha0;
      prog.va = M.va0;
      for (const r of prog.hist || []) {
        const k = r[2], p = r[4], y = r[5];
        if (!(k >= 2) || typeof p !== 'number') continue;
        const g = 1 / k;
        const s = clamp((p - g) / (1 - g), 0.02, 0.98);
        const m = Math.log(s / (1 - s)) + prog.alpha;
        this.learn(null, null, k, m, y);
      }
      // the aim's correction was making up for the old model's error about
      // you, which α now carries: start the correction afresh
      prog.ewma = null;
      prog.ctl = 0;
    }

    /* One answer's evidence, shared between the card (st.th), you (α), the
     * question type (β) and, for a card asked for the first time, how you do
     * on cards not yet asked (ν), in proportion to their variances; m is the
     * logit the answer was predicted with. Returns the predicted chance. */
    learn(st, type, k, m, y, first) {
      const prog = this.prog;
      const g = 1 / k, s = sigmoid(m), p = g + (1 - g) * s;
      const dP = (1 - g) * s * (1 - s);
      const G = y ? dP / Math.max(p, 1e-6) : -dP / Math.max(1 - p, 1e-6);   // d log L / d m
      const I = dP * dP / Math.max(p * (1 - p), 1e-6);                        // its Fisher information
      if (st && typeof st.v !== 'number') st.v = Math.max(M.vtMin, M.vt0 / (1 + 0.3 * (st.n || 0)));
      const vt = st ? st.v : 0, va = prog.va, vb = type ? prog.vb[type] : 0, vn = first ? prog.vn : 0;
      const den = 1 + I * (vt + va + vb + vn);
      const step = v => clamp(v * G / den, -1.5, 1.5);
      if (st) {
        st.th = clamp(st.th + step(vt), -4, 9);
        st.v = Math.max(M.vtMin, vt - vt * vt * I / den);
      }
      prog.alpha = clamp(prog.alpha + step(va), -4, 8);
      prog.va = Math.min(M.va0, va - va * va * I / den + M.qa);
      if (type) {
        prog.beta[type] = clamp(prog.beta[type] + step(vb), -3, 3);
        prog.vb[type] = Math.min(M.vb0, vb - vb * vb * I / den + M.qb);
      }
      if (first) {
        prog.nu = clamp(prog.nu + step(vn), -4, 4);
        prog.vn = Math.min(M.vn0, vn - vn * vn * I / den + M.qn);
      }
      return p;
    }

    setIndex(index) { this.index = index; }

    active(card) {
      const s = this.set;
      return s.types[card.type] !== false && s.areas[card.area] !== false && !this.prog.off[card.id]
        && !!this.index.byId[card.pid];
    }

    aim() {
      const t = this.set.target;
      const e = this.prog.ewma == null ? t : this.prog.ewma;
      const p = clamp(M.aimGain * (t - e), -M.aimSpan, M.aimSpan);
      return clamp(t + p + (this.prog.ctl || 0), 0.4, 0.97);
    }

    /* Spacing multiplier: a card is due when Δt ≥ τ·h. Recent results move
     * it, and so does a persistent shortfall (a positive integral term means
     * easier presentations alone have not been enough: repeat sooner). */
    tau() {
      const t = this.set.target;
      const e = this.prog.ewma == null ? t : this.prog.ewma;
      const short = Math.max(0, this.prog.ctl || 0);
      return clamp(1 + M.tauGain * (e - t) - 3 * short, M.tauMin, M.tauMax);
    }

    overdue(card, now) {
      const st = this.prog.cards[card.id];
      return Math.max(0, (now - st.last) / 60000) / st.h;
    }

    logit(card, k, dMean, now) {
      const st = this.prog.cards[card.id];
      // a card never asked: the prior for a card plus how you do on those
      // (ν), and nothing to forget yet
      const th = st ? st.th : M.th0 + (this.prog.nu || 0);
      const forget = st ? M.fMax * (1 - 1 / Math.sqrt(1 + Math.max(0, (now - st.last) / 60000) / st.h)) : 0;
      return th + (this.prog.alpha || 0) + (this.prog.beta[card.type] || 0) -
        M.gamma * Math.log(k - 1) - dMean - forget;
    }

    predict(card, k, dMean, now) {
      if (k <= 1) return 1;
      const g = 1 / k;
      return g + (1 - g) * sigmoid(this.logit(card, k, dMean, now));
    }

    /* The cards next() chooses among. `due`: cards due for review, most
     * overdue (Δt/h) first, the first of them a little randomised. `fresh`:
     * cards not yet asked — the next one in the introduction order when new
     * cards start as one-option lessons, otherwise the next one of each
     * question type, for next() to pick the one pitched nearest the aim.
     * `early`: the rest, nearest to coming due first; `ready`, those at least
     * halfway there. Never one of the last few shown, and preferably not from
     * the paper just shown. */
    pools(now) {
      const cards = this.index.cards.filter(c => this.active(c) && !this.skip.has(c.id));
      const recent = this.prog.recent.slice(-M.gap);
      const blocked = new Set(recent);
      const lastPid = recent.length ? (recent[recent.length - 1].split('|')[0]) : null;
      const tau = this.tau();
      const due = [], early = [], fresh = [], freshType = new Set();
      for (const c of cards) {
        if (!this.prog.cards[c.id]) {
          if (this.set.lessonFirst ? !fresh.length : !freshType.has(c.type)) { fresh.push(c); freshType.add(c.type); }
          continue;
        }
        if (blocked.has(c.id)) continue;
        const u = this.overdue(c, now);
        if (u >= tau) due.push({ c, u });
        else early.push({ c, u });
      }
      const notLast = arr => {
        const other = arr.filter(e => e.c.pid !== lastPid);
        return other.length ? other.concat(arr.filter(e => e.c.pid === lastPid)) : arr;
      };
      due.sort((a, b) => b.u - a.u);
      early.sort((a, b) => b.u - a.u);
      const ordered = notLast(due);
      if (ordered.length) {
        const top = ordered.slice(0, 3);
        const w = [0.6, 0.25, 0.15].slice(0, top.length);
        let r = this.rng() * w.reduce((a, b) => a + b, 0), first = 0;
        for (let i = 0; i < top.length; i++) { if ((r -= w[i]) <= 0) { first = i; break; } }
        ordered.unshift(ordered.splice(first, 1)[0]);
      }
      const late = notLast(early);
      let oldest = null;
      if (!ordered.length && !fresh.length && !late.length) {
        // every card asked so far was shown in the last few questions
        let bestQ = Infinity;
        for (const c of cards) {
          const st = this.prog.cards[c.id];
          if (st && st.lastQ < bestQ) { oldest = c; bestQ = st.lastQ; }
        }
      }
      return { due: ordered.map(e => e.c), fresh, early: late.map(e => e.c),
               ready: late.filter(e => e.u >= tau / 2).map(e => e.c), oldest };
    }

    pickCard(now) {
      const p = this.pools(now);
      return p.due[0] || p.fresh[0] || p.early[0] || p.oldest || null;
    }

    /* A question for `card`: options and presentation chosen so the predicted
     * accuracy is nearest the aim. `kCap` lets the layout ask for fewer
     * options when they will not fit on screen. */
    plan(card, now, kCap) {
      const cand = candidates(this.index, card);
      if (!cand) return null;
      // a card's first showing is a one-option lesson only if asked for;
      // otherwise it is a question like any other, pitched at the aim
      const intro = !this.prog.cards[card.id] && !!this.set.lessonFirst;
      const counts = cand.tiers.map(t => t.length);
      const avail = counts.reduce((a, b) => a + b, 0);
      const kMax = Math.min(kCap || M.kMax, this.set.maxOptions || M.kMax, avail + 1, M.kMax);
      if (intro) {
        return { card, intro: true, k: 1, tier: 0, p: 1, options: [Object.assign({ correct: true }, cand.correct)] };
      }
      // nothing to set against it (a pool emptied by filters or edits): pass
      // over the card for the rest of this visit rather than ask a 1-option review
      if (kMax < 2) { this.skip.add(card.id); return null; }
      const aim = this.aim();
      let best = null, hardest = null, easiest = null;
      for (let k = 2; k <= kMax; k++) {
        for (let s = 0; s < counts.length; s++) {
          const comp = composition(counts, k - 1, s);
          if (comp.short) continue;
          const p = this.predict(card, k, comp.mean, now);
          // nearest the aim; among near-ties, closer distractors, then fewer options
          const score = Math.abs(p - aim) - 0.004 * s + 0.002 * k;
          if (!best || score < best.score) best = { k, s, p, comp, score };
          if (!hardest || p < hardest.p) hardest = { k, s, p, comp };
          if (!easiest || p > easiest.p) easiest = { k, s, p, comp };
        }
      }
      if (!best) { this.skip.add(card.id); return null; }
      // A new card that even its easiest question would likely beat you on is
      // shown as a lesson first (its answer alone) instead of a guess.
      if (!this.prog.cards[card.id] && easiest.p < aim - M.lessonMargin) {
        return { card, intro: true, k: 1, tier: 0, p: 1, options: [Object.assign({ correct: true }, cand.correct)] };
      }
      // Even the hardest presentation too easy: ask that one. (Near certainty
      // every presentation's chance is squeezed toward 1 — 0.999 with two
      // options, 0.996 with six close ones — and the tie-breaks above would
      // otherwise pick the fewest options.)
      if (hardest.p > aim + 0.03) best = hardest;
      const picks = [];
      for (const [t, m] of best.comp.took) {
        const pool = shuffle(cand.tiers[t].slice(), this.rng);
        // the answer repeats a title word: have one wrong answer repeat one too
        // (from the closest tier that has one), so matching words against the
        // title does not single the answer out
        if (cand.cue && !picks.some(o => o.cue)) {
          const i = pool.findIndex(o => o.cue);
          if (i > 0) pool.unshift(pool.splice(i, 1)[0]);
        }
        picks.push(...pool.slice(0, m).map(o => Object.assign({ tier: t }, o)));
      }
      let dMean = best.comp.mean, p = best.p, tier = best.comp.took.length ? best.comp.took[0][0] : best.s;
      if (cand.cue && picks.length && !picks.some(o => o.cue)) {
        // no tier used had one: bring one in from another, in place of the
        // easiest pick, and predict from the options actually shown
        const key = o => o.kind + ':' + (o.pid || '') + '/' + (o.tid || '') + ':' + (o.text || '');
        const have = new Set(picks.map(key));
        let alt = null;
        for (let t = cand.tiers.length - 1; t >= 0 && !alt; t--) {
          const pool = shuffle(cand.tiers[t].filter(o => o.cue && !have.has(key(o))), this.rng);
          if (pool.length) alt = Object.assign({ tier: t }, pool[0]);
        }
        if (alt) {
          let at = 0;
          picks.forEach((o, i) => { if (o.tier < picks[at].tier) at = i; });
          picks[at] = alt;
          dMean = picks.reduce((a, o) => a + M.delta[o.tier], 0) / picks.length;
          p = this.predict(card, best.k, dMean, now);
          tier = Math.max(...picks.map(o => o.tier));
        }
      }
      const options = shuffle([Object.assign({ correct: true }, cand.correct), ...picks], this.rng);
      // `tier` is the tier the distractors actually came from (an empty close
      // tier falls back to same-area ones, and the label should say so);
      // `sat`, whether even the hardest presentation was predicted above the aim
      const sat = p > aim + 0.03 ? 'hard' : null;
      return { card, intro: false, k: best.k, tier, sat, dMean, p, options };
    }

    /* The next question. Due cards first. A question predicted well above
     * the aim even at its hardest — more options and closer distractors
     * cannot bring it down, as for a card you saw minutes ago or know cold —
     * is passed over for one that can be pitched at the aim; it comes back
     * once some forgetting makes it a test again. New cards are weighed
     * together: of the next one of each type, the one pitched nearest the
     * aim is asked, so the kinds of question that are hard for you come in
     * first. If everything is too easy, the hardest new card is asked (else
     * the first review passed over). */
    next(now, kCap) {
      const aim = this.aim();
      const pool = this.pools(now);
      const tooEasy = q => !q.intro && q.p > aim + M.passMargin;
      let fallback = null, passed = 0;
      // a review, or the card nearest to due: asked unless too easy (then
      // kept as the fallback); at most passMax are planned
      const ask = card => {
        if (passed >= M.passMax) return null;
        const q = this.plan(card, now, kCap);
        if (!q) return null;
        if (!tooEasy(q)) return q;
        if (!fallback) fallback = q;
        passed++;
        return null;
      };
      // a persistent shortfall with nothing due: review what is nearly due
      // before adding more
      const short = !pool.due.length && (this.prog.ctl || 0) > M.ctlMax / 2;
      const first = short && pool.fresh.length ? pool.ready.slice(0, 1) : pool.due.slice(0, 1);
      for (const card of first) { const q = ask(card); if (q) return q; }
      const plans = pool.fresh.map(c => this.plan(c, now, kCap)).filter(Boolean);
      const fit = plans.filter(q => !tooEasy(q)).sort((a, b) => Math.abs(a.p - aim) - Math.abs(b.p - aim));
      if (fit.length) return fit[0];
      const rest = pool.due.slice(1).concat(pool.early.filter(c => !first.includes(c)));
      for (const card of rest.slice(0, 60)) { const q = ask(card); if (q) return q; }
      // everything too easy: a new card rather than one more easy review, or
      // a run of right answers would keep you on the cards already seen
      if (plans.length) return plans.sort((a, b) => a.p - b.p)[0];
      if (!fallback && pool.oldest) fallback = this.plan(pool.oldest, now, kCap);
      return fallback;
    }

    snapshot(card) {
      const st = this.prog.cards[card.id];
      return JSON.stringify({ st: st || null, beta: this.prog.beta[card.type], vb: this.prog.vb[card.type],
        alpha: this.prog.alpha, va: this.prog.va, nu: this.prog.nu, vn: this.prog.vn,
        ewma: this.prog.ewma, ctl: this.prog.ctl || 0,
        q: this.prog.q, recent: this.prog.recent.slice(), hist: this.prog.hist.slice() });
    }

    restore(card, snap) {
      const s = JSON.parse(snap);
      if (s.st) this.prog.cards[card.id] = s.st; else delete this.prog.cards[card.id];
      this.prog.beta[card.type] = s.beta;
      if (typeof s.vb === 'number') this.prog.vb[card.type] = s.vb;
      if (typeof s.alpha === 'number') { this.prog.alpha = s.alpha; this.prog.va = s.va; }
      if (typeof s.nu === 'number') { this.prog.nu = s.nu; this.prog.vn = s.vn; }
      this.prog.ewma = s.ewma;
      this.prog.ctl = s.ctl;
      this.prog.q = s.q;
      this.prog.recent = s.recent;
      this.prog.hist = s.hist;   // whole history: at the 400-entry cap a length cannot undo the trim
    }

    /* Record an answer. Returns {correct, p}. */
    answer(question, correct, now) {
      const card = question.card;
      const prog = this.prog;
      prog.q++;
      prog.recent.push(card.id);
      if (prog.recent.length > 24) prog.recent.splice(0, prog.recent.length - 24);
      if (question.intro) {
        if (!prog.cards[card.id]) {
          prog.cards[card.id] = { n: 0, ok: 0, lapses: 0, th: M.th0, h: M.h0, last: now, lastQ: prog.q, intro: now };
        } else {
          Object.assign(prog.cards[card.id], { last: now, lastQ: prog.q });
        }
        return { correct: true, p: 1 };
      }
      const first = !prog.cards[card.id];
      const m = this.logit(card, question.k, question.dMean, now);   // before the card has a state
      if (first) prog.cards[card.id] = { n: 0, ok: 0, lapses: 0, th: M.th0 + (prog.nu || 0), h: M.h0, last: now, lastQ: prog.q, intro: now };
      const st = prog.cards[card.id];
      const k = question.k;
      const y = correct ? 1 : 0;
      const p = this.learn(st, card.type, k, m, y, first);
      const dt = Math.max(0, (now - st.last) / 60000);
      if (y) {
        const hard = 0.5 + 0.5 * Math.min(1, (M.gamma * Math.log(k - 1) + (question.dMean || 0)) / DMAX);
        // right at first sight: known from before the quiz, so a fully
        // spaced success, from a longer starting half-life
        if (first) st.h = M.hKnown;
        st.h = Math.min(M.hMax, st.h * (1 + 2.5 * hard * (first ? 1 : Math.sqrt(Math.min(1, dt / st.h)))));
        st.ok++;
      } else {
        st.h = Math.max(M.hMin, st.h * 0.35);
        st.lapses++;
      }
      st.n++;
      st.last = now;
      st.lastQ = prog.q;
      const e0 = prog.ewma == null ? this.set.target : prog.ewma;
      prog.ewma = e0 + M.ewmaRate * (y - e0);
      // Anti-windup: when even the hardest presentation was predicted easier
      // than the aim, a right answer says nothing the controller can act on;
      // integrating it would leave the aim far too low once the cards change.
      if (!(question.sat === 'hard' && y)) {
        prog.ctl = clamp((prog.ctl || 0) + M.ctlRate * (this.set.target - y), M.ctlMin, M.ctlMax);
      }
      prog.hist.push([now, card.type, k, question.tier, Math.round(p * 1000) / 1000, y]);
      if (prog.hist.length > 400) prog.hist.splice(0, prog.hist.length - 400);
      return { correct: !!y, p };
    }

    stats(now) {
      const cards = this.index.cards.filter(c => this.active(c));
      let seen = 0, solid = 0, due = 0;
      const tau = this.tau();
      for (const c of cards) {
        const st = this.prog.cards[c.id];
        if (!st) continue;
        seen++;
        if (st.h >= 60 * 24) solid++;
        if (this.overdue(c, now) >= tau) due++;
      }
      const h = this.prog.hist.slice(-50);
      const acc = h.length ? h.reduce((a, r) => a + r[5], 0) / h.length : null;
      return { total: cards.length, seen, solid, due, recent: h.length, acc };
    }
  }

  return {
    TYPES, TYPE_LABEL, AREAS, AREA_LABEL, MODEL: M,
    norm, authorString, shortAuthors, surname, personKey, authorsKey, labelKeys, maskSegments, jaccard, stable,
    mergePapers, merge3, buildIndex, thesisText, findTerm, findExample, optionContent, candidates,
    composition, freshProgress, defaultSettings, Engine,
  };
});
