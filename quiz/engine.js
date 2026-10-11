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
 * Each turn takes the most overdue due card (else a new card, else the card
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

  function words(s) { return new Set(norm(s).split(' ').filter(w => w.length > 2)); }
  function jaccard(a, b) {
    const A = words(a), B = words(b);
    if (!A.size || !B.size) return 0;
    let n = 0;
    for (const w of A) if (B.has(w)) n++;
    return n / (A.size + B.size - n);
  }

  const escRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  /* Split `text` into [{t: string, blank: bool}] with every needle (and its
   * plural) blanked, whole-word and case-insensitive. */
  function maskSegments(text, needles) {
    const list = [...new Set((needles || []).map(n => String(n || '').trim()).filter(n => n.length > 1))]
      .sort((a, b) => b.length - a.length);
    text = String(text || '');
    if (!list.length) return [{ t: text, blank: false }];
    const re = new RegExp('(?<![\\p{L}\\p{N}])(?:' + list.map(escRe).join('|') +
      ')(?:s|es)?(?![\\p{L}\\p{N}])', 'giu');
    const out = [];
    let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) out.push({ t: text.slice(last, m.index), blank: false });
      out.push({ t: m[0], blank: true });
      last = m.index + m[0].length;
      if (m[0].length === 0) re.lastIndex++;
    }
    if (last < text.length) out.push({ t: text.slice(last), blank: false });
    return out;
  }

  // ------------------------------------------------------------- content index

  /* Merge shipped papers with the user's edits: an edited or added paper
   * replaces the shipped one wholesale; a removed id drops it. */
  function mergePapers(base, edits) {
    const e = edits || {};
    const over = e.papers || {};
    const removed = new Set(e.removed || []);
    const out = [];
    const seen = new Set();
    for (const p of base) {
      if (removed.has(p.id)) continue;
      out.push(over[p.id] ? Object.assign({}, over[p.id], { _edited: true }) : p);
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
      for (const q of index.papers) {
        if (q.id === p.id || avoid.has(q.id) || (q.avoid || []).includes(p.id)) continue;
        const qt = thesisText(q);
        if (!qt || jaccard(qt, text) > 0.6) continue;
        const o = { kind: 'thesis', pid: q.id };
        const sameAuthor = first && norm(surname((q.authors || [])[0])) === first;
        if (q.topic === topic || sameAuthor) tiers[2].push(o);
        else if (q.area === area) tiers[1].push(o);
        else tiers[0].push(o);
      }
      return { correct: { kind: 'thesis', pid: p.id }, tiers };
    }

    if (card.type === 'd') {
      const t = findTerm(p, card.ref);
      if (!t) return null;
      const tn = norm(t.term);
      for (const q of index.papers) {
        for (const u of q.terms || []) {
          if (!u || !u.definition || !u.term) continue;
          if (q.id === p.id && u.id === t.id) continue;
          if (jaccard(u.definition, t.definition) > 0.6) continue;
          const o = { kind: 'def', pid: q.id, tid: u.id };
          if (q.id === p.id || norm(u.term) === tn || q.topic === topic) tiers[2].push(o);
          else if (q.area === area) tiers[1].push(o);
          else tiers[0].push(o);
        }
      }
      return { correct: { kind: 'def', pid: p.id, tid: t.id }, tiers };
    }

    if (card.type === 'x') {
      const x = findExample(p, card.ref);
      if (!x) return null;
      const right = norm(x.phenomenon);
      // `avoid` lists names that would also be a right answer for this example
      const avoid = new Set((x.avoid || []).map(norm));
      const clash = k => !k || k === right || k.includes(right) || right.includes(k) || avoid.has(k);
      // Never offer another name for the same thing: anything this paper's
      // own examples are labelled with, or the same topic's labels unless the
      // example's own near-miss list vouches for them.
      const sameTopic = new Set();
      for (const [k, e] of index.phenPool) if (e.topics.has(topic)) sameTopic.add(k);
      const used = new Set();
      for (const n of x.near || []) {
        const k = norm(n);
        if (clash(k) || used.has(k)) continue;
        used.add(k);
        tiers[2].push({ kind: 'phen', text: n });
      }
      for (const [k, e] of index.phenPool) {
        if (clash(k) || used.has(k) || sameTopic.has(k)) continue;
        used.add(k);
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
     * card; then the card nearest to coming due. Never one of the last few
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
        picks.push(...pool.slice(0, m).map(o => Object.assign({ tier: t }, o)));
      }
      const options = shuffle([Object.assign({ correct: true }, cand.correct), ...picks], this.rng);
      return { card, intro: false, k: best.k, tier: best.s, dMean: best.comp.mean, p: best.p, options };
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
        q: this.prog.q, recent: this.prog.recent.slice(), hlen: this.prog.hist.length });
    }

    restore(card, snap) {
      const s = JSON.parse(snap);
      if (s.st) this.prog.cards[card.id] = s.st; else delete this.prog.cards[card.id];
      this.prog.beta[card.type] = s.beta;
      this.prog.ewma = s.ewma;
      this.prog.ctl = s.ctl;
      this.prog.q = s.q;
      this.prog.recent = s.recent;
      this.prog.hist.length = s.hlen;
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
      prog.ewma = prog.ewma == null ? y : prog.ewma + M.ewmaRate * (y - prog.ewma);
      prog.ctl = clamp((prog.ctl || 0) + M.ctlRate * (this.set.target - y), M.ctlMin, M.ctlMax);
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
    norm, authorString, shortAuthors, surname, personKey, authorsKey, maskSegments, jaccard,
    mergePapers, buildIndex, thesisText, findTerm, findExample, optionContent, candidates,
    composition, freshProgress, defaultSettings, Engine,
  };
});
