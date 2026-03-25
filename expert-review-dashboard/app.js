/**
 * FAQ Chatbot Quality Evaluation — Expert Review App
 * Pure client-side SPA with sessionStorage persistence.
 */
(function () {
    'use strict';

    // ── Configuration ──
    const EMAILS = ['andrew.p.legear@ul.ie', 'andrew.legear@horizon-globex.ie'];
    const DATASET_PATH = '../evaluation-data/dataset.json';

    // ── State ──
    const DEFAULT_SAMPLE_SIZE = 50;
    let dataset = null;
    let reviewerId = '';
    let currentIndex = 0;
    let ratings = {}; // { itemId: { accuracy, completeness, helpfulness, hallucination, comment } }
    let sampleOrder = []; // indices into dataset.items, randomised
    let sampleSize = DEFAULT_SAMPLE_SIZE;

    // ── DOM refs ──
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // ── Storage helpers ──
    function storageKey() { return `eval-reviewer-${reviewerId}`; }

    function saveState() {
        const state = { reviewerId, currentIndex, ratings, sampleOrder, sampleSize };
        localStorage.setItem(storageKey(), JSON.stringify(state));
    }

    function loadState(id) {
        const raw = localStorage.getItem(`eval-reviewer-${id}`);
        if (!raw) return null;
        try { return JSON.parse(raw); } catch { return null; }
    }

    // ── Seeded shuffle (Fisher-Yates) ──
    function shuffleArray(arr) {
        const a = arr.slice();
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    // ── Dataset loading ──
    async function loadDataset() {
        try {
            const resp = await fetch(DATASET_PATH);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            dataset = await resp.json();
            return true;
        } catch (e) {
            alert(`Failed to load dataset from ${DATASET_PATH}\n\nMake sure dataset.json exists in the evaluation-data/ folder.\n\nError: ${e.message}`);
            return false;
        }
    }

    // ── Login ──
    async function handleLogin() {
        const idInput = $('#reviewer-id');
        const id = idInput.value.trim();
        if (!id) {
            idInput.focus();
            idInput.style.borderColor = '#ef4444';
            return;
        }

        reviewerId = id;
        const ok = await loadDataset();
        if (!ok) return;

        // Restore previous state if any
        const saved = loadState(id);
        if (saved && saved.sampleOrder && saved.sampleOrder.length > 0) {
            ratings = saved.ratings || {};
            currentIndex = saved.currentIndex || 0;
            sampleOrder = saved.sampleOrder;
            sampleSize = saved.sampleSize || saved.sampleOrder.length;
        } else {
            ratings = {};
            currentIndex = 0;
            sampleSize = parseInt($('#sample-size').value) || DEFAULT_SAMPLE_SIZE;
            // Build randomised sample
            const allIndices = Array.from({ length: dataset.items.length }, (_, i) => i);
            const shuffled = shuffleArray(allIndices);
            sampleOrder = shuffled.slice(0, Math.min(sampleSize, dataset.items.length));
        }

        showScreen('eval-screen');
        $('#reviewer-label').textContent = `Reviewer: ${reviewerId}`;
        renderItem();
    }

    // ── Screen management ──
    function showScreen(id) {
        $$('.screen').forEach(s => s.classList.add('hidden'));
        $(`#${id}`).classList.remove('hidden');
    }

    // ── Render current item ──
    function renderItem() {
        const datasetIndex = sampleOrder[currentIndex];
        const item = dataset.items[datasetIndex];

        $('#item-number').textContent = `Question ${currentIndex + 1} of ${sampleOrder.length}`;
        $('#panel-question').textContent = item.question;
        $('#panel-truth').textContent = item.ground_truth;
        $('#panel-response').textContent = item.chatbot_response;

        // Update progress — count rated items within the sample
        const sampleItemIds = new Set(sampleOrder.map(i => dataset.items[i].id));
        const ratedCount = Object.keys(ratings).filter(id => {
            const r = ratings[id];
            return sampleItemIds.has(parseInt(id)) && r.accuracy && r.completeness && r.helpfulness;
        }).length;
        const pct = Math.round((ratedCount / sampleOrder.length) * 100);
        $('#progress-fill').style.width = `${pct}%`;
        $('#progress-text').textContent = `${ratedCount}/${sampleOrder.length} rated`;

        // Restore ratings for this item
        const r = ratings[item.id] || {};
        $$('input[name="accuracy"]').forEach(el => { el.checked = (el.value == r.accuracy); });
        $$('input[name="completeness"]').forEach(el => { el.checked = (el.value == r.completeness); });
        $$('input[name="helpfulness"]').forEach(el => { el.checked = (el.value == r.helpfulness); });
        $('#hallucination-check').checked = !!r.hallucination;
        $('#comment-box').value = r.comment || '';

        // Nav button states
        $('#btn-prev').disabled = currentIndex === 0;
        $('#btn-next').textContent = currentIndex === items.length - 1 ? 'Summary' : 'Next →';
    }

    // ── Save current item ratings ──
    function saveCurrentRatings() {
        const datasetIndex = sampleOrder[currentIndex];
        const item = dataset.items[datasetIndex];
        const acc = document.querySelector('input[name="accuracy"]:checked');
        const comp = document.querySelector('input[name="completeness"]:checked');
        const help = document.querySelector('input[name="helpfulness"]:checked');

        ratings[item.id] = {
            accuracy: acc ? parseInt(acc.value) : null,
            completeness: comp ? parseInt(comp.value) : null,
            helpfulness: help ? parseInt(help.value) : null,
            hallucination: $('#hallucination-check').checked,
            comment: $('#comment-box').value.trim() || null,
        };
        saveState();
    }

    // ── Navigation ──
    function goNext() {
        saveCurrentRatings();
        if (currentIndex < sampleOrder.length - 1) {
            currentIndex++;
            renderItem();
            window.scrollTo(0, 0);
        } else {
            showSummary();
        }
    }

    function goPrev() {
        saveCurrentRatings();
        if (currentIndex > 0) {
            currentIndex--;
            renderItem();
            window.scrollTo(0, 0);
        }
    }

    // ── Summary ──
    function showSummary() {
        saveCurrentRatings();
        showScreen('summary-screen');

        // Only show items in the sample
        const sampleItems = sampleOrder.map(i => dataset.items[i]);
        const total = sampleItems.length;
        let ratedCount = 0;
        let accSum = 0, compSum = 0, helpSum = 0, hallCount = 0, accN = 0;

        sampleItems.forEach(item => {
            const r = ratings[item.id];
            if (r && r.accuracy && r.completeness && r.helpfulness) {
                ratedCount++;
                accSum += r.accuracy; compSum += r.completeness; helpSum += r.helpfulness;
                accN++;
            }
            if (r && r.hallucination) hallCount++;
        });

        // Stats
        $('#summary-stats').innerHTML = `
            <div class="stat-box"><div class="stat-value">${ratedCount}/${total}</div><div class="stat-label">Items Rated</div></div>
            <div class="stat-box"><div class="stat-value">${accN ? (accSum / accN).toFixed(1) : '—'}</div><div class="stat-label">Mean Accuracy</div></div>
            <div class="stat-box"><div class="stat-value">${accN ? (compSum / accN).toFixed(1) : '—'}</div><div class="stat-label">Mean Completeness</div></div>
            <div class="stat-box"><div class="stat-value">${accN ? (helpSum / accN).toFixed(1) : '—'}</div><div class="stat-label">Mean Helpfulness</div></div>
            <div class="stat-box"><div class="stat-value">${hallCount}</div><div class="stat-label">Hallucinations Flagged</div></div>
        `;

        // Warning
        const warn = $('#incomplete-warning');
        if (ratedCount < total) {
            warn.classList.remove('hidden');
            warn.innerHTML = `<strong>Note:</strong> ${ratedCount} of ${total} sampled items rated. You can export partial results at any time.`;
        } else {
            warn.classList.add('hidden');
        }

        // Table — only sample items
        let html = '<table class="summary-table"><thead><tr><th>#</th><th>Question</th><th>Acc</th><th>Comp</th><th>Help</th><th>Hall</th><th>Comment</th></tr></thead><tbody>';
        sampleItems.forEach((item, i) => {
            const r = ratings[item.id] || {};
            const complete = r.accuracy && r.completeness && r.helpfulness;
            const cls = complete ? '' : ' class="incomplete"';
            const q = item.question.length > 60 ? item.question.substring(0, 60) + '…' : item.question;
            html += `<tr${cls}>
                <td>${i + 1}</td>
                <td class="q-cell" title="${escapeHtml(item.question)}">${escapeHtml(q)}</td>
                <td>${r.accuracy || '—'}</td>
                <td>${r.completeness || '—'}</td>
                <td>${r.helpfulness || '—'}</td>
                <td>${r.hallucination ? '⚠️' : '—'}</td>
                <td class="q-cell">${escapeHtml(r.comment || '')}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        $('#summary-table-container').innerHTML = html;
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Export ──
    function buildExportData() {
        // Only export items in the sample that were actually rated
        const sampleItems = sampleOrder.map(i => dataset.items[i]);
        const exportRatings = sampleItems
            .filter(item => ratings[item.id])
            .map(item => {
                const r = ratings[item.id];
                return {
                    item_id: item.id,
                    question: item.question,
                    accuracy: r.accuracy || null,
                    completeness: r.completeness || null,
                    helpfulness: r.helpfulness || null,
                    hallucination: !!r.hallucination,
                    comment: r.comment || null,
                };
            });
        return {
            reviewer_id: reviewerId,
            completed_at: new Date().toISOString(),
            dataset_metadata: dataset.metadata,
            sample_size: sampleOrder.length,
            total_items: dataset.items.length,
            items_rated: exportRatings.length,
            sample_item_ids: sampleOrder.map(i => dataset.items[i].id),
            ratings: exportRatings,
        };
    }

    function exportEmail() {
        const data = buildExportData();
        const json = JSON.stringify(data, null, 2);
        const subject = encodeURIComponent(`FAQ Evaluation Results - ${reviewerId}`);
        const body = encodeURIComponent(`Expert review results from ${reviewerId}:\n\n${json}`);
        const mailto = `mailto:${EMAILS.join(',')}?subject=${subject}&body=${body}`;

        // mailto URLs have a practical limit of ~2000 chars
        if (mailto.length > 2000) {
            alert('Results are too large for email. Use "Download Results as JSON" instead.');
            exportDownload();
            return;
        }
        window.location.href = mailto;
    }

    function exportDownload() {
        const data = buildExportData();
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `evaluation-${reviewerId}-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // ── Event listeners ──
    function bindEvents() {
        // Login
        $('#btn-start').addEventListener('click', handleLogin);
        $('#reviewer-id').addEventListener('keydown', (e) => { if (e.key === 'Enter') handleLogin(); });

        // Nav
        $('#btn-next').addEventListener('click', goNext);
        $('#btn-prev').addEventListener('click', goPrev);
        $('#btn-summary').addEventListener('click', () => { saveCurrentRatings(); showSummary(); });
        $('#btn-finish').addEventListener('click', () => { saveCurrentRatings(); showSummary(); });
        $('#btn-back-eval').addEventListener('click', () => { showScreen('eval-screen'); renderItem(); });

        // Export
        $('#btn-export-email').addEventListener('click', exportEmail);
        $('#btn-export-download').addEventListener('click', exportDownload);

        // Auto-save on any rating change
        $$('input[name="accuracy"], input[name="completeness"], input[name="helpfulness"]').forEach(el => {
            el.addEventListener('change', () => saveCurrentRatings());
        });
        $('#hallucination-check').addEventListener('change', () => saveCurrentRatings());
        $('#comment-box').addEventListener('input', debounce(() => saveCurrentRatings(), 500));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Only on eval screen
            if ($('#eval-screen').classList.contains('hidden')) return;
            // Don't intercept if typing in textarea
            if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;

            if (e.key === 'ArrowRight') { e.preventDefault(); goNext(); }
            else if (e.key === 'ArrowLeft') { e.preventDefault(); goPrev(); }
            else if (e.key >= '1' && e.key <= '5') {
                // Quick-set accuracy
                const radio = document.querySelector(`input[name="accuracy"][value="${e.key}"]`);
                if (radio) { radio.checked = true; saveCurrentRatings(); }
            }
        });

        // Summary table row click -> navigate to that item
        $('#summary-table-container').addEventListener('click', (e) => {
            const tr = e.target.closest('tr');
            if (!tr || tr.parentElement.tagName === 'THEAD') return;
            const idx = tr.rowIndex - 1; // -1 for thead row
            if (idx >= 0 && idx < sampleOrder.length) {
                currentIndex = idx;
                showScreen('eval-screen');
                renderItem();
            }
        });
    }

    function debounce(fn, ms) {
        let timer;
        return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
    }

    // ── Init ──
    document.addEventListener('DOMContentLoaded', bindEvents);
})();
