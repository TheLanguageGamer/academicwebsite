/* Paper quiz — content model and adaptive engine (no DOM; runs in Node for tests).
 *
 * Cards. Each paper yields an author card (a), a thesis card (t), one card per
 * term (d, "definition") and one per example (x). A card's id is built from the
 * paper id and the term/example id, so edits to the wording keep its history.
 *
 * Model. For card c shown with k options whose distractors have mean closeness
 * cost D_s, after Δt minutes since it was last seen:
 *
 *     P(correct) = 1/k + (1 − 1/k) · σ(θ_c + β_type − γ·ln(k−1) − D_s − F(Δt/h_c))
 *     F(r) = F_max · (1 − (1 + r)^(−1/2))
 *
 * F is forgetting: about 0.9 logits at one half-life, levelling off at F_max.
 * It is bounded because this is recognition among options, not recall — a
 * paper you once knew does not sink to chance — and an unbounded curve made
 * every long-overdue card look forgotten (predictions ran 15 points low).
 *
 * θ_c is how well you discriminate this card's answer, h_c its memory
 * half-life (minutes), β_type a per-question-type calibration shared by all
 * cards. After each answer θ_c and β_type take a gradient step on the log
 * likelihood (so they settle where predictions match what you actually do),
 * and h_c grows on a success (more for a harder, better-spaced success) and
 * shrinks on a miss.
 *
 * The three difficulty levers the quiz pulls:
 *   - number of options k (1 the first time a card is seen, then 2–6);
 *   - closeness of the wrong options (tier 0 random, 1 same area, 2 same topic
 *     or hand-picked near misses);
 *   - spacing: a card comes due once the time since you last saw it reaches
 *     τ·h_c. τ is 1 at target and moves with your results (down to 0.3 when
 *     you are below target, so reviews come sooner; up to 1.6 above it).
 * Each turn takes the most overdue due card (else a new card — or, while you
 * are persistently below target, a nearly due one first — else the card
 * nearest to due) and the (k, closeness) whose predicted accuracy is nearest
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
    delta: [0, 0.9, 1.8],   // cost of a distractor by closeness tier
    fMax: 3.0,              // forgetting ceiling (logits)
    th0: 1.5,               // θ of a card just introduced
    h0: 2,                  // half-life (min) of a card just introduced
    hMin: 1.5,
    hMax: 60 * 24 * 365 * 2,
    tauGain: 2.5, tauMin: 0.3, tauMax: 1.6, // spacing multiplier around 1
    betaRate: 0.12,         // step of the per-type calibration
    gap: 4,                 // a card is not repeated within this many questions
    kMax: 6,
    ewmaRate: 0.08,
    aimGain: 0.5, aimSpan: 0.08,          // proportional term
    ctlRate: 0.012, ctlMin: -0.3, ctlMax: 0.15,   // integral term
  };
  const DMAX = M.gamma * Math.log(M.kMax - 1) + M.delta[2];

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
   * picked for 'Common Ground'). A question that names the title could then be
   * answered by matching words, so plan() makes sure at least one wrong answer
   * repeats one of them too. */
  const CUE_STOP = new Set(['that', 'this', 'with', 'from', 'what', 'when', 'which', 'some', 'about',
    'into', 'their', 'there', 'toward', 'towards', 'does', 'have']);
  const cueWords = memo(s => new Set(norm(s).split(' ').filter(w => w.length > 3 && !CUE_STOP.has(w)).map(stem)));
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
    const tiers = [[], [], []];
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
        if (e.topics.has(topic) || shares || (e.areas.has(area) && nearYear)) tiers[2].push(o);
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
        if (q.topic === topic || sameAuthor) tiers[2].push(o);
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
          if (q.id === p.id || norm(u.term) === tn || q.topic === topic) tiers[2].push(o);
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
        tiers[2].push({ kind: 'phen', text: n });
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
        if (e.areas.has(area)) tiers[1].push({ kind: 'phen', text: e.text });
        else tiers[0].push({ kind: 'phen', text: e.text });
      }
      return { correct: { kind: 'phen', text: x.phenomenon }, tiers };
    }
    return null;
  }

  /* Mean closeness cost of n distractors drawn from tier s first, then the
   * easier tiers, then (if the pool is that thin) the harder ones. */
  function composition(counts, n, s) {
    const order = [];
    for (let t = s; t >= 0; t--) order.push(t);
    for (let t = s + 1; t <= 2; t++) order.push(t);
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
      ewma: null, ctl: 0, recent: [], hist: [], off: {},
    };
  }

  function defaultSettings() {
    return {
      target: 0.8, maxOptions: 6, showYear: true, autoAdvance: false,
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
      for (const t of TYPES) if (typeof this.prog.beta[t] !== 'number') this.prog.beta[t] = 0;
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
      const dt = Math.max(0, (now - st.last) / 60000);
      const forget = M.fMax * (1 - 1 / Math.sqrt(1 + dt / st.h));
      return st.th + (this.prog.beta[card.type] || 0) - M.gamma * Math.log(k - 1) - dMean - forget;
    }

    predict(card, k, dMean, now) {
      if (k <= 1) return 1;
      const g = 1 / k;
      return g + (1 - g) * sigmoid(this.logit(card, k, dMean, now));
    }

    /* Which card next. Due cards first, most overdue (Δt/h) first; then a new
     * card (unless you are persistently short and a card is at least halfway
     * to due); then the card nearest to coming due. Never one of the last few
     * shown, and preferably not from the paper just shown. */
    pickCard(now) {
      const cards = this.index.cards.filter(c => this.active(c) && !this.skip.has(c.id));
      const recent = this.prog.recent.slice(-M.gap);
      const blocked = new Set(recent);
      const lastPid = recent.length ? (recent[recent.length - 1].split('|')[0]) : null;
      const tau = this.tau();
      const due = [], early = [];
      let fresh = null;
      for (const c of cards) {
        if (!this.prog.cards[c.id]) {
          if (!fresh) fresh = c;
          continue;
        }
        if (blocked.has(c.id)) continue;
        const u = this.overdue(c, now);
        if (u >= tau) due.push({ c, u });
        else early.push({ c, u });
      }
      const notLast = arr => {
        const other = arr.filter(e => e.c.pid !== lastPid);
        return other.length ? other : arr;
      };
      if (due.length) {
        due.sort((a, b) => b.u - a.u);
        const top = notLast(due).slice(0, 3);
        const w = [0.6, 0.25, 0.15].slice(0, top.length);
        let r = this.rng() * w.reduce((a, b) => a + b, 0);
        for (let i = 0; i < top.length; i++) { if ((r -= w[i]) <= 0) return top[i].c; }
        return top[0].c;
      }
      // a persistent shortfall: review what is nearly due before adding more
      if (fresh && (this.prog.ctl || 0) > M.ctlMax / 2 && early.length) {
        early.sort((a, b) => b.u - a.u);
        const e = notLast(early)[0];
        if (e.u >= tau / 2) return e.c;
      }
      if (fresh) return fresh;
      if (early.length) {
        early.sort((a, b) => b.u - a.u);
        return notLast(early)[0].c;
      }
      // every introduced card was shown in the last few questions
      let best = null, bestQ = Infinity;
      for (const c of cards) {
        const st = this.prog.cards[c.id];
        if (st && st.lastQ < bestQ) { best = c; bestQ = st.lastQ; }
      }
      return best;
    }

    /* A question for `card`: options and presentation chosen so the predicted
     * accuracy is nearest the aim. `kCap` lets the layout ask for fewer
     * options when they will not fit on screen. */
    plan(card, now, kCap) {
      const cand = candidates(this.index, card);
      if (!cand) return null;
      const intro = !this.prog.cards[card.id];
      const counts = cand.tiers.map(t => t.length);
      const avail = counts[0] + counts[1] + counts[2];
      const kMax = Math.min(kCap || M.kMax, this.set.maxOptions || M.kMax, avail + 1, M.kMax);
      if (intro) {
        return { card, intro: true, k: 1, tier: 0, p: 1, options: [Object.assign({ correct: true }, cand.correct)] };
      }
      // nothing to set against it (a pool emptied by filters or edits): pass
      // over the card for the rest of this visit rather than ask a 1-option review
      if (kMax < 2) { this.skip.add(card.id); return null; }
      const aim = this.aim();
      let best = null;
      for (let k = 2; k <= kMax; k++) {
        for (let s = 0; s <= 2; s++) {
          const comp = composition(counts, k - 1, s);
          if (comp.short) continue;
          const p = this.predict(card, k, comp.mean, now);
          // nearest the aim; among near-ties, closer distractors, then fewer options
          const score = Math.abs(p - aim) - 0.004 * s + 0.002 * k;
          if (!best || score < best.score) best = { k, s, p, comp, score };
        }
      }
      if (!best) { this.skip.add(card.id); return null; }
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
      const options = shuffle([Object.assign({ correct: true }, cand.correct), ...picks], this.rng);
      // the tier the distractors actually came from (an empty close tier
      // falls back to same-area ones, and the label should say so), and
      // whether even the hardest presentation was predicted above the aim
      const tier = best.comp.took.length ? best.comp.took[0][0] : best.s;
      const sat = best.p > aim + 0.03 ? 'hard' : null;
      return { card, intro: false, k: best.k, tier, sat, dMean: best.comp.mean, p: best.p, options };
    }

    next(now, kCap) {
      for (let tries = 0; tries < 50; tries++) {
        const card = this.pickCard(now);
        if (!card) return null;
        const q = this.plan(card, now, kCap);
        if (q) return q;
      }
      return null;
    }

    snapshot(card) {
      const st = this.prog.cards[card.id];
      return JSON.stringify({ st: st || null, beta: this.prog.beta[card.type], ewma: this.prog.ewma, ctl: this.prog.ctl || 0,
        q: this.prog.q, recent: this.prog.recent.slice(), hist: this.prog.hist.slice() });
    }

    restore(card, snap) {
      const s = JSON.parse(snap);
      if (s.st) this.prog.cards[card.id] = s.st; else delete this.prog.cards[card.id];
      this.prog.beta[card.type] = s.beta;
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
      const st = prog.cards[card.id];
      const k = question.k;
      const g = 1 / k;
      const m = this.logit(card, k, question.dMean, now);
      const s = sigmoid(m);
      const p = g + (1 - g) * s;
      const y = correct ? 1 : 0;
      // d log L / d m for P = g + (1−g)σ(m)
      const dP = (1 - g) * s * (1 - s);
      const grad = y ? dP / Math.max(p, 1e-6) : -dP / Math.max(1 - p, 1e-6);
      const K = Math.max(0.3, 0.9 / Math.sqrt(1 + 0.3 * st.n));
      st.th = clamp(st.th + clamp(K * grad, -1.5, 1.5), -4, 9);
      prog.beta[card.type] = clamp((prog.beta[card.type] || 0) + M.betaRate * grad, -3, 3);
      const dt = Math.max(0, (now - st.last) / 60000);
      if (y) {
        const hard = 0.5 + 0.5 * Math.min(1, (M.gamma * Math.log(k - 1) + (question.dMean || 0)) / DMAX);
        st.h = Math.min(M.hMax, st.h * (1 + 2.5 * hard * Math.sqrt(Math.min(1, dt / st.h))));
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
