import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Search, Loader2, BookOpen, Database, AlertCircle, Download,
  CheckCircle2, Layers, Quote, ShieldCheck, ShieldAlert
} from 'lucide-react';

// ---- Env-driven API base (set VITE_API_URL in production) ----
const API_BASE = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:8000';

interface Paper {
  pmid: string;
  title: string;
  authors: string;
  journal?: string;
  pubdate: string;
  source_type?: string;
}

interface Extraction {
  pmid: string;
  title?: string;
  dataset?: string;
  sample_size?: string;
  validation?: string;
  model?: string;
  metrics?: string;
  key_findings?: string;
}

interface ResearchResponse {
  run_id: string;
  query: string;
  status: string;
  review_status: string;
  keywords: string[];
  sub_questions: string[];
  pmids: string[];
  papers: Paper[];
  extractions: Extraction[];
  dataset_comparison: Array<{ dataset_family: string; papers: Array<any> }>;
  method_comparison: Array<{ pmid: string; model: string; validation: string; key_findings: string }>;
  metric_summary: Array<{ pmid: string; reported: string; numeric_percentages: number[] }>;
  research_gaps: string[];
  references: Paper[];
  synthesis: string;
}

const STAGES = ['Searching', 'Screening', 'Extracting', 'Comparing', 'Synthesizing', 'Completed'];

export default function App() {
  const [query, setQuery] = useState('Review recent research on deep-learning methods for EEG seizure detection and compare datasets, models, validation methods, and evaluation metrics.');
  const [loading, setLoading] = useState(false);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [data, setData] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'review' | 'papers' | 'matrix' | 'comparison' | 'gaps' | 'references'>('review');

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const newRunId = crypto.randomUUID();
    setLoading(true);
    setError(null);
    setData(null);
    setCurrentStageIndex(0);

    try {
      const response = await fetch(`${API_BASE}/api/review/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, run_id: newRunId }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split('\n\n');
        buffer = chunks.pop() || '';

        for (const chunk of chunks) {
          const line = chunk.trim();
          if (!line.startsWith('data:')) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;

          let evt: any;
          try { evt = JSON.parse(jsonStr); } catch { continue; }

          if (evt.type === 'stage') {
            const idx = STAGES.indexOf(evt.stage);
            if (idx >= 0) setCurrentStageIndex(idx);
          } else if (evt.type === 'result') {
            if (evt.data?.run_id && evt.data.run_id !== newRunId) return;
            setData(evt.data);
            setCurrentStageIndex(STAGES.length - 1);
          } else if (evt.type === 'error') {
            setError(evt.message || 'Pipeline execution failed.');
          }
        }
      }
    } catch (err: any) {
      setError(err.message || 'Failed to complete research pipeline execution.');
    } finally {
      setLoading(false);
    }
  };

  const downloadMarkdown = () => {
    if (!data?.synthesis) return;
    const header = `# Literature Review\n\n**Query:** ${data.query}\n\n**Review Status:** ${data.review_status}\n\n---\n\n`;
    const refBlock = data.references?.length
      ? `\n\n---\n\n## References\n\n${data.references.map((r, i) => `${i + 1}. ${r.authors}. ${r.title}. ${r.journal || ''} (${r.pubdate}). PMID: ${r.pmid}`).join('\n')}`
      : '';
    const blob = new Blob([header + data.synthesis + refBlock], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `literature_review_${Date.now()}.md`;
    a.click();
  };

  const reviewApproved = data?.review_status?.startsWith('Approved');

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-600/20 text-blue-400 rounded-lg border border-blue-500/30">
              <BookOpen className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-300 bg-clip-text text-transparent">
                Medical Literature AI Assistant
              </h1>
              <p className="text-xs text-slate-400">
                Autonomous LangGraph Agent Engine • Groq GPT-OSS-20B • Live SSE Streaming
              </p>
            </div>
          </div>
          <span className="text-xs px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" /> API Connected
          </span>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full space-y-8">
        <section className="bg-slate-950/50 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
          <form onSubmit={handleSearch} className="flex gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter clinical prompt..."
                className="w-full bg-slate-900 border border-slate-700/80 rounded-xl pl-12 pr-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium px-8 py-3.5 rounded-xl transition flex items-center gap-2 shadow-lg shadow-blue-600/20 cursor-pointer"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Execute Review'}
            </button>
          </form>
        </section>

        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-3 text-red-400">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p className="text-sm">{error}</p>
          </div>
        )}

        {loading && (
          <div className="bg-slate-950/40 border border-slate-800 rounded-2xl p-8 text-center space-y-6">
            <Loader2 className="w-10 h-10 animate-spin text-blue-400 mx-auto" />
            <h3 className="text-lg font-medium text-slate-200">
              Literature Review Status: <span className="text-blue-400">{STAGES[currentStageIndex]}</span>
            </h3>
            <div className="flex justify-center gap-2 flex-wrap">
              {STAGES.map((stage, idx) => (
                <span
                  key={stage}
                  className={`px-3 py-1.5 rounded-full text-xs border transition ${
                    idx <= currentStageIndex
                      ? 'bg-blue-600/20 border-blue-500/40 text-blue-300 font-medium'
                      : 'bg-slate-900 border-slate-800 text-slate-500'
                  }`}
                >
                  {stage}
                </span>
              ))}
            </div>
          </div>
        )}

        {data && (
          <div className="space-y-6">
            {/* Reviewer verdict banner */}
            <div
              className={`p-4 rounded-xl border flex items-start gap-3 text-sm ${
                reviewApproved
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                  : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
              }`}
            >
              {reviewApproved ? (
                <ShieldCheck className="w-5 h-5 flex-shrink-0 mt-0.5" />
              ) : (
                <ShieldAlert className="w-5 h-5 flex-shrink-0 mt-0.5" />
              )}
              <div>
                <p className="font-medium">Reviewer Agent Verdict</p>
                <p className="text-xs opacity-90">{data.review_status}</p>
              </div>
            </div>

            {/* Query / keywords / sub-questions summary */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 bg-slate-950/40 border border-slate-800 rounded-xl">
                <p className="text-xs uppercase text-slate-500 mb-2">Generated PubMed Query</p>
                <code className="text-xs text-blue-300 break-words">{data.keywords?.[0] || '—'}</code>
              </div>
              <div className="p-4 bg-slate-950/40 border border-slate-800 rounded-xl">
                <p className="text-xs uppercase text-slate-500 mb-2">Sub-questions</p>
                <ul className="text-xs text-slate-300 space-y-1 list-disc pl-4">
                  {data.sub_questions?.map((q, i) => <li key={i}>{q}</li>)}
                </ul>
              </div>
            </div>

            {/* Nav tabs */}
            <div className="flex border-b border-slate-800 justify-between items-center overflow-x-auto">
              <div className="flex gap-4">
                {[
                  { id: 'review', label: 'Literature Review' },
                  { id: 'papers', label: `Papers (${data.papers?.length || 0})` },
                  { id: 'matrix', label: `Evidence Matrix (${data.extractions?.length || 0})` },
                  { id: 'comparison', label: 'Comparisons' },
                  { id: 'gaps', label: `Research Gaps (${data.research_gaps?.length || 0})` },
                  { id: 'references', label: `References (${data.references?.length || 0})` },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`pb-3 text-sm font-medium border-b-2 transition cursor-pointer whitespace-nowrap ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-400'
                        : 'border-transparent text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {activeTab === 'review' && data.synthesis && (
                <button
                  onClick={downloadMarkdown}
                  className="mb-2 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 flex items-center gap-1.5 transition cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" /> Export Markdown
                </button>
              )}
            </div>

            {/* Literature Review */}
            {activeTab === 'review' && (
              <div className="bg-slate-950/40 border border-slate-800 rounded-2xl p-8 text-slate-200 leading-relaxed space-y-4
                              [&_table]:w-full [&_table]:border-collapse [&_table]:my-6
                              [&_th]:border [&_th]:border-slate-700 [&_th]:bg-slate-800/90 [&_th]:p-3 [&_th]:text-left [&_th]:text-slate-200 [&_th]:font-semibold
                              [&_td]:border [&_td]:border-slate-800 [&_td]:p-3 [&_td]:text-slate-300 [&_tr:nth-child(even)]:bg-slate-900/40">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.synthesis}</ReactMarkdown>
              </div>
            )}

            {/* Papers */}
            {activeTab === 'papers' && (
              <div className="grid gap-4">
                {data.papers?.length ? data.papers.map((paper, i) => (
                  <div key={i} className="p-5 bg-slate-950/40 border border-slate-800 rounded-xl space-y-2">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <span className="text-xs font-mono bg-blue-500/10 text-blue-400 px-2.5 py-1 rounded border border-blue-500/20">
                        PMID: {paper.pmid}
                      </span>
                      <div className="flex items-center gap-2">
                        {paper.source_type && (
                          <span className="text-xs px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
                            {paper.source_type}
                          </span>
                        )}
                        <span className="text-xs text-slate-400">{paper.pubdate}</span>
                      </div>
                    </div>
                    <h4 className="font-semibold text-slate-200">{paper.title}</h4>
                    <p className="text-xs text-slate-400">
                      {paper.authors} {paper.journal && `• ${paper.journal}`}
                    </p>
                  </div>
                )) : (
                  <div className="p-8 text-center text-slate-400 bg-slate-950/40 border border-slate-800 rounded-xl">
                    No publications retrieved for this query.
                  </div>
                )}
              </div>
            )}

            {/* Evidence Matrix */}
            {activeTab === 'matrix' && (
              <div className="bg-slate-950/40 border border-slate-800 rounded-2xl overflow-x-auto">
                {data.extractions?.length ? (
                  <table className="w-full text-left text-sm text-slate-300">
                    <thead className="bg-slate-900 border-b border-slate-800 text-slate-400 text-xs uppercase">
                      <tr>
                        <th className="p-4">PMID</th>
                        <th className="p-4">Dataset</th>
                        <th className="p-4">N</th>
                        <th className="p-4">Validation</th>
                        <th className="p-4">Model</th>
                        <th className="p-4">Metrics</th>
                        <th className="p-4">Key Findings</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {data.extractions.map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-900/50">
                          <td className="p-4 font-mono text-blue-400">{row.pmid}</td>
                          <td className="p-4">{row.dataset || 'N/A'}</td>
                          <td className="p-4">{row.sample_size || 'N/A'}</td>
                          <td className="p-4">{row.validation || 'N/A'}</td>
                          <td className="p-4 font-medium text-slate-200">{row.model || 'N/A'}</td>
                          <td className="p-4 text-emerald-400">{row.metrics || 'N/A'}</td>
                          <td className="p-4 text-xs max-w-xs leading-relaxed">{row.key_findings || 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="p-8 text-center text-slate-400">No extracted evidence matrix available.</div>
                )}
              </div>
            )}

            {/* Comparisons */}
            {activeTab === 'comparison' && (
              <div className="grid gap-6">
                <div className="p-6 bg-slate-950/40 border border-slate-800 rounded-2xl space-y-4">
                  <h3 className="font-semibold text-blue-400 flex items-center gap-2">
                    <Database className="w-4 h-4" /> Dataset Comparisons
                  </h3>
                  {data.dataset_comparison?.length ? (
                    <div className="space-y-3">
                      {data.dataset_comparison.map((group, i) => (
                        <div key={i} className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                          <p className="text-sm font-medium text-slate-200 mb-2">{group.dataset_family}</p>
                          <ul className="space-y-1">
                            {group.papers.map((p, j) => (
                              <li key={j} className="text-xs text-slate-400">
                                <span className="font-mono text-blue-400">PMID {p.pmid}</span>: {p.sample_size || 'N/A'} — <span className="text-slate-300">{p.validation || 'N/A'}</span> — <span className="text-emerald-400">{p.metrics || 'N/A'}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500">No datasets available for comparison.</p>
                  )}
                </div>

                <div className="p-6 bg-slate-950/40 border border-slate-800 rounded-2xl space-y-4">
                  <h3 className="font-semibold text-indigo-400 flex items-center gap-2">
                    <Layers className="w-4 h-4" /> Method Comparisons
                  </h3>
                  {data.method_comparison?.length ? (
                    <ul className="space-y-3">
                      {data.method_comparison.map((m, i) => (
                        <li key={i} className="p-3 bg-slate-900 border border-slate-800 rounded-lg text-xs">
                          <span className="font-mono text-indigo-400">PMID {m.pmid}</span>: <strong className="text-slate-200">{m.model}</strong> — {m.key_findings}
                          <div className="text-slate-500 mt-1">Validation: {m.validation}</div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-slate-500">No methodologies available for comparison.</p>
                  )}
                </div>
              </div>
            )}

            {/* Research Gaps */}
            {activeTab === 'gaps' && (
              <div className="p-6 bg-slate-950/40 border border-slate-800 rounded-2xl space-y-4">
                <h3 className="font-semibold text-amber-400">Identified Research Gaps & Open Challenges</h3>
                {data.research_gaps?.length ? (
                  <ul className="space-y-3">
                    {data.research_gaps.map((gap, i) => (
                      <li key={i} className="p-4 bg-slate-900 border border-slate-800 rounded-xl text-sm text-slate-300 flex items-start gap-3">
                        <span className="px-2 py-0.5 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded text-xs font-semibold">Gap #{i + 1}</span>
                        {gap}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-slate-500">No research gaps identified for this query context.</p>
                )}
              </div>
            )}

            {/* References */}
            {activeTab === 'references' && (
              <div className="p-6 bg-slate-950/40 border border-slate-800 rounded-2xl space-y-4">
                <h3 className="font-semibold text-slate-200 flex items-center gap-2">
                  <Quote className="w-4 h-4" /> References ({data.references?.length || 0})
                </h3>
                {data.references?.length ? (
                  <ol className="space-y-3 list-decimal list-inside">
                    {data.references.map((r, i) => (
                      <li key={i} className="p-3 bg-slate-900 border border-slate-800 rounded-lg text-sm">
                        <span className="text-slate-300">{r.authors}. </span>
                        <span className="text-slate-100 font-medium">{r.title}. </span>
                        <span className="text-slate-400 italic">{r.journal || ''} </span>
                        <span className="text-slate-400">({r.pubdate}). </span>
                        <a
                          href={`https://pubmed.ncbi.nlm.nih.gov/${r.pmid}/`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-blue-400 hover:text-blue-300 font-mono text-xs"
                        >
                          PMID: {r.pmid}
                        </a>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="text-xs text-slate-500">No references available.</p>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}